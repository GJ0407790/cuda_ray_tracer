#include <iostream>

#include "config.hpp"
#include "config_utils.cuh"
#include "draw.cuh"
#include "parse.hpp"
#include "libpng.h"

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

int main(int argc, char* argv[]){
	if(argc != 2)
	{
		std::cerr << "Use case: make run file=your/file.txt" << endl;
		exit(1);
	}

	StlConfig host_stl_config;
	//parse the inputs into host config
	parseInput(argv, host_stl_config);
	std::cout << "After parseInput" << std::endl;

	RawConfig host_raw_config;
	// init from stl config
	initRawConfigFromStl(host_stl_config, host_raw_config);
	std::cout << "After initRawConfigFromStl" << std::endl;
	
	// device allocations
	copyRawConfigToDevice(host_raw_config);
	CUDA_CHECK(cudaPeekAtLastError());
	std::cout << "After copyRawConfigToDevice" << std::endl;

	// copy config to gpu
	RawConfig* d_raw_config;
	CUDA_CHECK(cudaMalloc(&d_raw_config, sizeof(RawConfig)));
	CUDA_CHECK(cudaMemcpy(d_raw_config, &host_raw_config, sizeof(RawConfig), cudaMemcpyHostToDevice));

	// create the rgba array in gpu
	RGBA* d_image;
	CUDA_CHECK(cudaMalloc(&d_image, host_stl_config.width * host_stl_config.height * sizeof(RGBA)));

	render(d_image, host_stl_config.width, host_stl_config.height, host_stl_config.aa, d_raw_config);
	CUDA_CHECK(cudaDeviceSynchronize());
	std::cout << "After render" << std::endl;

	// create the image
	Image img(host_stl_config.width, host_stl_config.height);
	// copy the rendered image from gpu to cpu
	CUDA_CHECK(cudaMemcpy(img[0], d_image, host_stl_config.width * host_stl_config.height * sizeof(RGBA), cudaMemcpyDeviceToHost));

	std::string output_path = host_stl_config.filename;
	img.save(output_path.c_str());

	// free host memory
	freeStlConfig(host_stl_config);

	// // free gpu memory
	CUDA_CHECK(cudaFree(d_image));
	CUDA_CHECK(cudaFree(d_raw_config));
	// freeRawConfigDeviceMemory(host_raw_config);
	// CUDA_CHECK(cudaPeekAtLastError());
}
