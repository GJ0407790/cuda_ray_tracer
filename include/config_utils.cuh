#ifndef CONFIG_UTILS_CUH
#define CONFIG_UTILS_CUH

#include <cuda_runtime.h>
#include "config.hpp" 

// -------------- Host-Side Functions --------------

// 1) Initialize a RawConfig from a StlConfig (on host)
void initRawConfigFromStl(const StlConfig& host_stl, RawConfig& out_rc);

// 2) Copy all device arrays in RawConfig to device memory
void copyRawConfigToDevice(RawConfig& rc);

// 3) Recursively deep-copy an Object to device
Object* deepCopyObjectToDevice(const Object* host_obj);

// 4) Recursively deep-copy a BVH to device
BVH* copyBVHToDevice(BVH* host_bvh);

// -------------- Freeing Device Memory -------------

// 5) Free the device memory allocated for a BVH subtree
void freeBVHOnDevice(BVH* d_bvh);

// 6) Free the device memory allocated for an Object
void freeDeviceObject(Object* d_obj);

// 7) Free all device arrays (and BVH) in RawConfig
void freeRawConfigDeviceMemory(RawConfig& rc);

// -------------- Freeing Host Memory -------------
// 8) Free the host memory allocated for a stl conifg
void freeStlConfig(StlConfig& stl);

#endif // CONFIG_UTILS_CUH
