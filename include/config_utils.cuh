#ifndef CONFIG_UTILS_CUH
#define CONFIG_UTILS_CUH

#include <cuda_runtime.h>
#include "config.hpp" // Includes the updated StlConfig and RawConfig

// -------------- Host-Side Functions --------------

// Initializes a host-side RawConfig structure based on a host-side StlConfig.
// Device pointers in out_rc_host_mirror will be nullptr.
void initRawConfigFromStl(const StlConfig& host_stl, RawConfig& out_rc_host_mirror);

// Allocates device memory and copies data from StlConfig to the device.
// Fills the device pointer members (d_all_spheres, d_all_suns, etc.) in rc_with_device_ptrs.
void copyConfigDataToDevice(const StlConfig& host_stl, RawConfig& rc_with_device_ptrs);

// -------------- Freeing Device Memory -------------

// Frees all device memory pointed to by members of rc_with_device_ptrs.
void freeRawConfigDeviceMemory(RawConfig& rc_with_device_ptrs);

#endif // CONFIG_UTILS_CUH