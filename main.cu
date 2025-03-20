#include <iostream>

#include "include/config.hpp"
#include "include/draw.hpp"
#include "include/parse.hpp"

using std::cout;
using std::endl;

#define CUDA_CHECK(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error in " << __FILE__ << " at line "       \
                      << __LINE__ << " : " << cudaGetErrorString(err)      \
                      << std::endl;                                        \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }


__constant__ RawConfig config;

int main(int argc, char* argv[]){
	if(argc != 2)
	{
		std::cerr << "Use case: make run file=your/file.txt" << endl;
		exit(1);
	}

	StlConfig host_stl_config;
	//parse the inputs into host config
	parseInput(argv, host_stl_config);

	RawConfig host_raw_config(host_stl_config);

	// copy config to gpu constant memory
	CUDA_CHECK(cudaMemcpyToSymbol(config, &host_raw_config, sizeof(RawConfig)));

	// create the rgba array in gpu
	RGBA* d_image;
	CUDA_CHECK(cudaMalloc(&d_image, host_stl_config.width * host_stl_config.height * sizeof(RGBA)));

	render(d_image, host_stl_config.width, host_stl_config.height, host_stl_config.aa);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// create the image
	Image img(host_stl_config.width, host_stl_config.height);
	// copy the rendered image from gpu to cpu
	CUDA_CHECK(cudaMemcpy(img[0], d_image, host_stl_config.width * host_stl_config.height * sizeof(RGBA), cudaMemcpyDeviceToHost));

	std::string output_path = std::string(getenv("SLURM_TMPDIR")) + "/" + host_stl_config.filename;
	img.save(output_path.c_str());

	// free gpu memory
	CUDA_CHECK(cudaFree(d_image));
}
