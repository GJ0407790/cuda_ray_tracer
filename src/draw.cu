#include "draw.cuh"
#include "helper.cuh"
#include "bvh_traversal.cuh"

#include <math.h>

#define EPSILON 0.001f

__global__ void finalize_kernel(pixel_t* d_image, const float4* d_accum_buffer, const int img_width, const int img_height, const int aa) 
{
	// Use standard 1D grid-stride loop or simple index calculation
	const int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;
	const int total_pixels = img_width * img_height;

	if (pixel_idx >= total_pixels) {
		return;
	}

	// Read accumulated value
	float4 accumulated_rgba = d_accum_buffer[pixel_idx];

	// Calculate average color
	float inv_aa = 1.0f / (float)aa; // Ensure float division
	float avg_r = accumulated_rgba.x * inv_aa;
	float avg_g = accumulated_rgba.y * inv_aa;
	float avg_b = accumulated_rgba.z * inv_aa;
	float avg_a = accumulated_rgba.w * inv_aa;

	// Apply gamma correction and convert to unsigned char for final image
	// Ensure results are clamped to [0, 255]
	int r_final = (int)(fminf(fmaxf(RGBtosRGB(avg_r), 0.0f), 1.0f) * 255.0f + 0.5f); // Add 0.5 for rounding
	int g_final = (int)(fminf(fmaxf(RGBtosRGB(avg_g), 0.0f), 1.0f) * 255.0f + 0.5f);
	int b_final = (int)(fminf(fmaxf(RGBtosRGB(avg_b), 0.0f), 1.0f) * 255.0f + 0.5f);
	int a_final = (int)(fminf(fmaxf(avg_a, 0.0f), 1.0f) * 255.0f + 0.5f);         // Alpha usually not gamma corrected

	// Write to output image buffer
	int h = pixel_idx / img_width;
	int w = pixel_idx % img_width;
	int image_offset = h * img_width + w;

	d_image[image_offset].r = (unsigned char)r_final;
	d_image[image_offset].g = (unsigned char)g_final;
	d_image[image_offset].b = (unsigned char)b_final;
	d_image[image_offset].a = (unsigned char)a_final;
}

__global__ void render_kernel_atomic_aa(float4* d_accum_buffer, const int img_width, const int img_height, const int aa, RawConfig* config)
{
	// Calculate global thread ID
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int total_pixels = img_width * img_height;
    const int total_samples = total_pixels * aa;

	// Check if thread is out of bounds (for total samples)
	if (tid >= total_samples) {
		return; 
	}

	// Determine pixel index and sample index for this thread
	const int pixel_idx = tid / aa; 
	const int sample_idx_in_pixel = tid % aa; // Useful for stratified/correlated sampling if desired

	// Calculate pixel coordinates (w, h)
	const int w = pixel_idx % img_width; 
	const int h = pixel_idx / img_width; 

	// Setup unique RNG state for this thread (sample)
	curandState state;
	// Seed using global thread ID (tid) for uniqueness across all samples
	curand_init(1234 + pixel_idx, sample_idx_in_pixel, 0, &state); // Example seeding variation

	// Generate jittered coordinates within the pixel [w, w+1), [h, h+1)
	// Using simple uniform random jitter for now
	float jitter_x = randD(-0.5f, 0.5f, &state);
	float jitter_y = randD(-0.5f, 0.5f, &state);
	// Add stratified or other sampling patterns here if needed, using sample_idx_in_pixel

	float sample_w = (float)w + jitter_x;
	float sample_h = (float)h + jitter_y;

	// Shoot one primary ray for this sample
	RGBA rgba_sample = shootPrimaryRay(sample_w, sample_h, &state, config);

	atomicAdd(&d_accum_buffer[pixel_idx].x, rgba_sample.r);
	atomicAdd(&d_accum_buffer[pixel_idx].y, rgba_sample.g);
	atomicAdd(&d_accum_buffer[pixel_idx].z, rgba_sample.b);
	atomicAdd(&d_accum_buffer[pixel_idx].w, rgba_sample.a); // Accumulate alpha too
}

__global__ void render_kernel(pixel_t* d_image, const int img_width, const int img_height, const int aa, RawConfig* config)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid >= img_height * img_width)
	{
		return; // out of bounds
	}

	// setup rng state
	curandState state;
	curand_init(1234, tid, 0, &state);

	const int w = tid % img_width; // width
	const int h = tid / img_width; // height

	RGBA rgba;
	if (aa == 0)
	{
		rgba = shootPrimaryRay((float)w, (float)h, &state, config);
	}
	else
	{
		RGBA new_rgba;
		for(int i = 0; i < aa; i++) 
		{
			float new_w = w + randD(-0.5f, 0.5f, &state);
			float new_h = h + randD(-0.5f, 0.5f, &state);

			new_rgba = new_rgba + shootPrimaryRay(new_w, new_h, &state, config);
		}

		rgba = new_rgba.mean(aa);
	}

	d_image[h * img_width + w].r = RGBtosRGB(rgba.r) * 255;
	d_image[h * img_width + w].g = RGBtosRGB(rgba.g) * 255;
	d_image[h * img_width + w].b = RGBtosRGB(rgba.b) * 255;
	d_image[h * img_width + w].a = rgba.a * 255;
}

void render(pixel_t* d_image, const int img_width, const int img_height, const int aa, RawConfig* config)
{
	if (aa <= 1) { 
		constexpr int block_size = 128;
		int grid_size = (img_width * img_height - 1) / block_size + 1;

		render_kernel<<<grid_size, block_size>>>(d_image, img_width, img_height, aa, config);

		return;
	}

	// --- Atomic AA Path (assuming aa >= 1) ---

	// 1. Allocate accumulation buffer (using float4 for RGBA)
	float4* d_accum_buffer = nullptr;
	size_t accum_buffer_size = (size_t)img_width * img_height * sizeof(float4);
	cudaMalloc(&d_accum_buffer, accum_buffer_size);
	cudaMemset(d_accum_buffer, 0, accum_buffer_size); // Initialize sums to zero

	// 2. Launch render kernel (one thread per sample)
	constexpr int block_size = 128; // Or tune this
	// Total threads needed: width * height * aa (if aa=0 was handled, use max(1,aa))
	int total_threads = img_width * img_height * aa;
	int grid_size = (total_threads + block_size - 1) / block_size;

	printf("[DEBUG Render] Launching AA Kernel. Total Threads: %d, Grid: %d, Block: %d\n", total_threads, grid_size, block_size); // Debug
	render_kernel_atomic_aa<<<grid_size, block_size>>>(
		d_accum_buffer, 
		img_width, 
		img_height, 
		aa, // Pass effective_aa
		config
	);

	// 3. Launch finalize kernel (one thread per pixel)
	int finalize_total_threads = img_width * img_height;
	int finalize_grid_size = (finalize_total_threads + block_size - 1) / block_size;
	
  printf("[DEBUG Render] Launching Finalize Kernel. Total Threads: %d, Grid: %d, Block: %d\n", finalize_total_threads, finalize_grid_size, block_size); // Debug
	finalize_kernel<<<finalize_grid_size, block_size>>>(
		d_image,
		d_accum_buffer,
		img_width,
		img_height,
		aa // Pass effective_aa for averaging
	);

	// 4. Free accumulation buffer
	cudaFree(d_accum_buffer);

    // Optional: Synchronize if timing or subsequent steps depend on completion
    // cudaDeviceSynchronize(); 
}

// void render(pixel_t* d_image, const int img_width, const int img_height, const int aa, RawConfig* config)
// {
// 	constexpr int block_size = 128;
// 	int grid_size = (img_width * img_height - 1) / block_size + 1;

// 	render_kernel<<<grid_size, block_size>>>(d_image, img_width, img_height, aa, config);
// }

/**
 * @brief Shoots a primary ray into the scene, at pixel location (x,y).
 * @param x x coordinate of the pixel.
 * @param y y coordinate of the pixel.
 * @return RGBA the final pixel color value, in linear RGB color space.
 * @details This function shoots one primary ray for the pixel (x,y). It will 
 * 			calculate the color for the pixel with the nearest object the primary
 * 			ray hit, and the final color will be the blend of diffuse, reflect, 
 * 			refract and global illumination color. The latter three could create
 * 			more rays that will bounce in the scene.
 */
__device__ RGBA shootPrimaryRay(float x, float y, curandState* state, RawConfig* config){
	//create a ray
	Ray ray(x, y, state, config);
	
	//loop over all objects in the scene
	ObjectInfo obj = hitNearest(ray, config);

	RGBA color, diffuse, reflect, refract, gi_color;

	if(obj.isHit)
	{ //hit
		diffuse = diffuseLight(obj, state, config);
		reflect = reflectionLight(ray,obj, state, config);
		refract = refractionLight(ray,obj, state, config);
		gi_color = obj.mat.color * globalIllumination(obj, config->gi, state, config);

		//mix the colors
		color = obj.mat.shininess * reflect + 
						(RGB(1.0f, 1.0f, 1.0f) - obj.mat.shininess) * obj.mat.trans * refract + 
						(RGB(1.0f, 1.0f, 1.0f) - obj.mat.shininess) * 
						(RGB(1.0f, 1.0f, 1.0f) - obj.mat.trans) * (diffuse + gi_color);
		color.a = 1.0f;
	} 

	return color;
}

/**
 * @brief Gets the nearest object in the path of the ray.
 * @param ray The ray to trace.
 * @return ObjectInfo All the related info for the object, if any is hit.
 */
__device__ ObjectInfo hitNearest(Ray& ray, RawConfig* config)
{
	if(ray.bounce == 0) return ObjectInfo();

	// Initial t_max for the ray. Can be from a global scene extent or a large number.
	// If your Ray struct has a t_max member, use that. Otherwise, pass a large float.
	float initial_ray_t_max = float(INFINITY); // Or some other suitable large value

	// Call the LBVH traversal function
	ObjectInfo bvh_hit = traverse_lbvh(ray, config, initial_ray_t_max);
	ObjectInfo plane_hit = checkPlane(ray, false, config); 

	if (bvh_hit.isHit && plane_hit.isHit) 
	{
		return (bvh_hit.distance < plane_hit.distance) ? bvh_hit : plane_hit;
	} 
	else if (bvh_hit.isHit) 
	{
		return bvh_hit;
	} 
	else if (plane_hit.isHit) 
	{
		return plane_hit;
	}
	
	return ObjectInfo();
}

/**
 * @brief Get the diffuse light color,with shadow checking.
 * @param obj The objectInfo instance with the intersection information.
 * @return RGBA The RGBA linear color at the location.
 * @details The diffuse light color is checked by creating a shadow ray from the intersection
 * 			point and the normal. This ray is checked against all light sources, and the 
 * 			actual color is found with the lambert light model. If a object totally obstructs
 * 			a shadow ray, that ray does not contribute to the diffuse lighting.
 */
__device__ RGBA diffuseLight(const ObjectInfo& obj, curandState* state, RawConfig* config){
	RGBA color;
	vec3 normal = obj.normal;
	
	if(obj.mat.roughness > 0.0f)
	{
		normal = normal + vec3(standerdD(obj.mat.roughness, state),
													 standerdD(obj.mat.roughness, state),
													 standerdD(obj.mat.roughness, state));
	}
		
	normal = normal.normalize();

	for(int i = 0; i < config->num_sun; i++)
	{
		const auto& light = config->d_all_suns[i];
		//Create a shadow ray, check if path blocked
		Ray shadow_ray(obj.i_point + obj.normal*EPSILON, light.dir,1);
		auto sunInfo = hitNearest(shadow_ray, config);

		if(sunInfo.isHit)
		{
			continue;
		} 
		
		float lambert = fmax(dot(normal, light.dir.normalize()), 0.0f);
		color = color + getColorSun(lambert, obj.mat.color, light.color, config);
	}
	
	//iterate over all point lights(bulbs)
	for(int i = 0; i < config->num_bulbs; i++){
		const auto& light = config->d_all_bulbs[i];
		//Create a shadow ray, check if path blocked
		vec3 bulbDir = (light.point - obj.i_point);
		Ray shadow_ray(obj.i_point + obj.normal * EPSILON, bulbDir, 1);

		auto bulbInfo = hitNearest(shadow_ray, config);

		if (bulbInfo.isHit && bulbInfo.distance < bulbDir.length())
		{
			continue;
		}

		float lambert = fmax(dot(normal,bulbDir.normalize()), 0.0f);
		color = color + getColorBulb(lambert, obj.mat.color, light.color,bulbDir.length(), config);
	}

	return color;
}


/**
 * @brief Get the reflection light color.
 * @param ray The ray which caused the reflection.
 * @param obj The objectInfo instance with the intersection information.
 * @return RGBA The RGBA linear color at the location.
 * @details The reflection ray is calculated and checked against all objects in 
 * 			the scene. If the reflection ray did hit an object, a full light
 * 			calculation is performed to achieve the reflection lighting effect.
 */
__device__ RGBA reflectionLight(const Ray& ray,const ObjectInfo& obj, curandState* state, RawConfig* config){
	if(obj.mat.shininess == RGB(0.0f, 0.0f, 0.0f) || ray.bounce <= 0) return RGBA();

	vec3 normal = obj.normal;
	if(obj.mat.roughness > 0.0f)
	{
		normal = normal + vec3(standerdD(obj.mat.roughness, state),
													 standerdD(obj.mat.roughness, state),
													 standerdD(obj.mat.roughness, state));
	}

	normal = normal.normalize();
	vec3 reflect_dir = ray.dir - 2.0f * (dot(normal,ray.dir)) * normal;
	Ray second_ray(obj.i_point + obj.normal * EPSILON, reflect_dir, ray.bounce - 1);

	ObjectInfo second_obj = hitNearest(second_ray, config);

	RGB shine,trans;
	if(ray.bounce == 1)
	{
		shine = RGB(0.0f, 0.0f, 0.0f);
		trans = RGB(0.0f, 0.0f, 0.0f);
	}
	else
	{
		shine = second_obj.mat.shininess;
		trans = second_obj.mat.trans;
	}

	RGBA color, diffuse, reflect, refract;

	if(second_obj.isHit)
	{ //hit
		diffuse = diffuseLight(second_obj, state, config);
		reflect = reflectionLight(second_ray,second_obj, state, config);
		refract = refractionLight(ray, obj, state, config);

		color = shine * reflect + 
						(RGB(1.0f, 1.0f, 1.0f) - shine) * trans * refract + 
						(RGB(1.0f, 1.0f, 1.0f) - shine) * (RGB(1.0f, 1.0f, 1.0f) - trans) * diffuse;
					
	}
	else
	{
		 color = RGBA(0.0f, 0.0f, 0.0f, 1.0f);
	}

	return color;
}

/**
 * @brief Get the refraction light color.
 * @param ray The ray which caused the refraction.
 * @param obj The objectInfo instance with the intersection information.
 * @return RGBA The RGBA linear color at the location.
 * @details The refraction discriminant (k) is calculated and if k < 0, total 
 * 			internal refraction occur, which is treated as reflection. If not, the
 * 			refraction ray is calculated, and checked against all objects again(this 
 * 			currently only works for spheres and could be optimized). When the ray hits
 * 			the object again, the final refraction ray is calculated using the inverse
 * 			of the ior ratio during ray entrance. After the ray exits, full light calculation
 * 			is performed.
 * @note Currently this function only handles air-object intersection.
 * 		 It also assumes that total internal refraction will not happen inside
 * 		 the object itself, and the sphere-intersection could be optimized when
 * 	 	 the ray is inside the object.
 */
__device__ RGBA refractionLight(const Ray& ray, const ObjectInfo& obj, curandState* state, RawConfig* config){
	if(obj.mat.trans == RGB(0.0f, 0.0f, 0.0f) || ray.bounce <= 0) return RGBA();
	
	vec3 refract_dir;
	Ray inside_ray,final_ray;

	float ior = 1.0f / obj.mat.ior;
	vec3 dir = ray.dir;
	vec3 normal = obj.normal.normalize();
	point3 i_point = obj.i_point;
	int bounce = ray.bounce;

	float k = 1.0f - pow(ior, 2) * (1.0f - pow(dot(normal,dir),2));
	
	if(k < 0)
	{ //total internal refraction
		//use the reflection method instead
		refract_dir = dir - 2.0f * (dot(normal,dir)) * normal;
		final_ray = Ray(i_point + normal * EPSILON, refract_dir, --bounce);
	}
	else
	{ //refraction inside the object
		refract_dir = ior * dir - (ior * (dot(normal,dir)) + sqrt(k)) * normal;
		inside_ray = Ray(i_point - normal* 0.0001f, refract_dir, bounce);

		//The object that the light goes out,usually the same sphere
		ObjectInfo other_obj = hitNearest(inside_ray, config);
		
		normal = other_obj.normal.normalize();
		ior = other_obj.mat.ior;
		dir = inside_ray.dir;
		i_point = other_obj.i_point;

		k = 1.0f - ior * ior * (1.0f - pow(dot(normal,dir), 2));

		refract_dir = ior * dir - (ior * (dot(normal,dir)) + sqrt(k)) * normal;
		final_ray = Ray(i_point - normal * 0.0001f, refract_dir, --bounce);
	}

	ObjectInfo final_obj = hitNearest(final_ray, config);
	RGB shine,trans;
	
	if(bounce == 0)
	{
		shine = RGB(0.0f, 0.0f, 0.0f);
		trans = RGB(0.0f, 0.0f, 0.0f);
	}
	else
	{
		shine = final_obj.mat.shininess;
		trans = final_obj.mat.trans;
	}

	RGBA color,diffuse,reflect,refract;

	if(final_obj.isHit)
	{ //hit
		diffuse = diffuseLight(final_obj, state, config);
		reflect = reflectionLight(final_ray,final_obj, state, config);
		refract = refractionLight(final_ray,final_obj, state, config);
		//mix the colors
		color = shine * reflect + 
						(RGB(1.0f, 1.0f, 1.0f) - shine) * trans * refract + 
						(RGB(1.0f, 1.0f, 1.0f) - shine) * (RGB(1.0f, 1.0f, 1.0f) - trans) * diffuse;
	}
	else 
	{
		color = RGBA(0.0f, 0.0f, 0.0f, 1.0f);
	}

	return color;
}

/**
 * @brief Get the global illumination light color.
 * @param obj The objectInfo instance with the intersection information.
 * @param gi_bounce Remaining bounces for global illumination rays.
 * @return RGBA The RGBA linear color at the location.
 * @details A gi ray is created by selecting a new ray in the sphere at the intersection 
 * 			point. This gi ray, after intersection, can create more gi rays based
 * 			on the remaining gi bounce.
 * @deprecated This version of global illumination is too slow and does not work
 * 			   that well. Will be replaced by something better later.
 */
__device__ RGBA globalIllumination(const ObjectInfo& obj, int gi_bounce, curandState* state, RawConfig* config){
	if(config->gi == 0 || gi_bounce == 0) return RGBA(); //exit if global illumination disabled
	
	vec3 normal = obj.normal;
	point3 i_point = obj.i_point;
	//sample a point on the unit sphere, with the center being the intersection point
	vec3 gi_dir = (normal + spherePoint(state)).normalize();
	
	Ray gi_ray(i_point + normal * EPSILON, gi_dir, gi_bounce-1);
	ObjectInfo gi_obj = hitNearest(gi_ray, config);
	
	RGBA color,diffuse,reflect,refract,gi_color;
	
	if(gi_obj.isHit)
	{ //hit
		diffuse = diffuseLight(gi_obj, state, config);
		reflect = reflectionLight(gi_ray,gi_obj, state, config);
		refract = refractionLight(gi_ray,gi_obj, state, config);
		gi_color = gi_obj.mat.color * globalIllumination(gi_obj, gi_bounce - 1, state, config);
		
		//mix the colors
		color = gi_obj.mat.shininess * reflect + 
						(RGB(1.0f, 1.0f, 1.0f) - gi_obj.mat.shininess) * gi_obj.mat.trans * refract + 
						(RGB(1.0f, 1.0f, 1.0f) - gi_obj.mat.shininess) * (RGB(1.0f, 1.0f, 1.0f) - gi_obj.mat.trans)*(diffuse + gi_color);
		color.a = 1.0f;
	}
	
	return color;
}

/**
 * @brief Check if any planes are intersecting with the ray.
 * @param ray The ray to check against.
 * @param exit_early For shadow checking purposes, exit early if a plane is in the
 *        way, casting shadows. Do not set to true with bulb(light in scene).
 * @return ObjectInfo The objectInfo instance which contains all intersection informations.
 * @details The parametric distance t is calculated. If t < 0, this means that the ray
 * 			intersection is behind the ray origin, which means no intersection. Else, 
 * 			calculate the intersection point with t.
 *
 */
__device__ ObjectInfo checkPlane(Ray& ray, bool exit_early, RawConfig* config){
	float t_sol = float(INFINITY);
	point3 p_sol;
	vec3 nor;
	Materials mats;

	for(int i = 0; i < config->num_planes; i++)
	{
		const auto& plane = config->d_all_planes[i];

		float t = dot((plane.point - ray.eye), plane.nor) / (dot(ray.dir, plane.nor));
		
		if(t <= 1e-6f)
		{
			 continue;
		}
		
		point3 intersection_point = t * ray.dir + ray.eye;
		
		if(t < t_sol && t > EPSILON)
		{
			t_sol = t;
			p_sol = intersection_point;
			nor = (dot(plane.nor, ray.dir) < 0.0f) ? plane.nor : -plane.nor;
			mats = plane.mat;
		}
	}

	if(t_sol >= INT_MAX - 10)
	{
		return ObjectInfo();
	}

	return ObjectInfo(t_sol, p_sol, nor,mats); 
}

/**
 * @brief Get the Color of Sun (directional light)
 * @param lambert The lambert constant.
 * @param objColor The color of the object.
 * @param lightColor The color of the light.
 * @return RGB The linear RGB color, after blending.
 * @details This function applies the lambert constant to the light color
 * 			and get the correct color by blending the colors of the light
 * 			and the object. Also takes care of exposure.
 */
__device__ RGBA getColorSun(float lambert, RGB objColor, RGB lightColor, RawConfig* config)
{
	float r,g,b;

	r = objColor.r * (lightColor.r * lambert);
	g = objColor.g * (lightColor.g * lambert);
	b = objColor.b * (lightColor.b * lambert);

	return RGBA(setExpose(r, config), setExpose(g, config), setExpose(b, config), 0.0f);
}

/**
 * @brief Get the Color of Bulb (scene light)
 * @param lambert The lambert constant.
 * @param objColor The color of the object.
 * @param lightColor The color of the light.
 * @return RGB The linear RGB color, after blending.
 * @details This function applies the lambert constant to the light color
 * 			and get the correct color by blending the colors of the light
 * 			and the object, and applys light intensity falloff.
 * 			Also takes care of exposure.`
 */
__device__ RGBA getColorBulb(float lambert, RGB objColor, RGB lightColor, float t, RawConfig* config)
{
	float r,g,b;
	float i = 1.0f / pow(t, 2);

	r = objColor.r * (lightColor.r * lambert);
	g = objColor.g * (lightColor.g * lambert);
	b = objColor.b * (lightColor.b * lambert);

	return RGBA(setExpose(r, config) * i, setExpose(g, config) * i, setExpose(b, config) * i, 0.0f);
}
