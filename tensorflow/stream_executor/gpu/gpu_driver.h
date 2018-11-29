/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// ROCM userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_

#include <stddef.h>
#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"

namespace stream_executor {
namespace gpu {

// Identifies the memory space where an allocation resides. See
// GPUDriver::GetPointerMemorySpace().
enum class MemorySpace { kHost, kDevice };

// Returns a casual string, such as "host" for the provided memory space.
string MemorySpaceString(MemorySpace memory_space);

class GPUContext;

// GPUDriver contains wrappers for calls to the userspace library driver. It's
// useful to isolate these calls and put basic wrappers around them to separate
// userspace library driver behaviors from the rest of the program.
//
// At the moment it's simply used as a namespace.
//
// The calls log any specific errors internally and return whether the operation
// was successful to the caller.
//
// Thread safety: these functions should not be used from signal handlers.
class GPUDriver {
 public:
  // Wraps a call to hipInit with logging to help indicate what has gone wrong in
  // the case of failure. Safe to call multiple times; will be fast on all calls
  // after the first.
  static port::Status Init();

  // Creates a new ROCM stream associated with the given device via
  // hipStreamCreate.
  // stream is an outparam owned by the caller, must not be null.
  static bool CreateStream(GPUContext* context, GPUStreamHandle* stream);

  // Destroys a ROCM stream associated with the given context.
  // stream is owned by the caller, must not be null, and *stream is set to null
  // if the stream is successfully destroyed.
  static void DestroyStream(GPUContext* context, GPUStreamHandle* stream);

  // ROCM events can explicitly disable event TSC retrieval for some presumed
  // performance improvement if timing is unnecessary.
  enum class EventFlags { kDefault, kDisableTiming };

  // Creates a new event associated with the given context.
  // result is an outparam owned by the caller and must not be null.
  static port::Status CreateEvent(GPUContext* context, GPUEventHandle* result,
                                  EventFlags flags);

  // Destroys *event and turns it into a nullptr. event may not be null, but
  // *event may be, via hipEventDestroy
  static port::Status DestroyEvent(GPUContext* context, GPUEventHandle* event);

  // Allocates a GPU memory space of size bytes associated with the given
  // context via hipMemAlloc.
  static void* DeviceAllocate(GPUContext* context, uint64 bytes);

  // Deallocates a GPU memory space of size bytes associated with the given
  // context via hipMemFree.
  static void DeviceDeallocate(GPUContext* context, void* location);

  // Allocates a unified memory space of size bytes associated with the given
  // context (not yet supported in HIP)
  static void* UnifiedMemoryAllocate(GPUContext* context, uint64 bytes);

  // Deallocates a unified memory space of size bytes associated with the given
  // context (not yet supported in HIP)
  static void UnifiedMemoryDeallocate(GPUContext* context, void* location);

  // Allocates page-locked and ROCM-registered memory on the host via
  // hipMemAllocHost.
  static void* HostAllocate(GPUContext* context, uint64 bytes);

  // Deallocates a location created by HostAllocate, via hipMemFreeHost.
  static void HostDeallocate(GPUContext* context, void* location);

  // Registers a memory region at location of size bytes via hipMemHostRegister.
  static bool HostRegister(GPUContext* context, void* location, uint64 bytes);

  // Unregisters a memory region that was previously registered at location via
  // hipMemHostUnregister.
  //
  //
  // TODO(leary) verify an error will be returned if the location wasn't
  // previously registered.
  static bool HostUnregister(GPUContext* context, void* location);

  // Given a device ordinal, returns a device handle into the device outparam,
  // which must not be null.
  //
  // N.B. these device handles do not have a corresponding destroy function in
  // the ROCM driver API.
  static port::Status GetDevice(int device_ordinal, GPUDeviceHandle *device);

  // Given a device handle, returns the name reported by the driver for the
  // device.
  static bool GetDeviceName(GPUDeviceHandle device, string *name_out);

  // Given a device to create a context for, returns a context handle into the
  // context outparam, which must not be null.
  static port::Status CreateContext(int device_ordinal, GPUDeviceHandle device,
                                    const DeviceOptions& device_options,
                                    GPUContext** context);
  // Destroys the provided context via hipCtxDestroy.
  // Don't do this while clients could still be using the context, per the docs
  // bad things will happen.
  static void DestroyContext(GPUContext* context);

  // Queries the runtime for the specified attribute of the specified function.
  static bool FuncGetAttribute(GPUFunctionAttribute attribute,
                               GPUFunctionHandle function, int *attribute_value);

  // Sets the preferred cache configuration for the specified function.
  static bool FuncSetCacheConfig(GPUFunctionHandle function,
                                 GPUFuncCachePreference cache_config);

  // Gets the preferred shared memory bank configuration for the specified
  // CONTEXT (not function!), either default or four- or eight-byte bank size.
  static port::StatusOr<GPUSharedMemConfig> ContextGetSharedMemConfig(
      GPUContext* context);

  // Sets the preferred shared memory bank configuration for the specified
  // CONTEXT (not function!), either default or four- or eight-byte bank size.
  static port::Status ContextSetSharedMemConfig(
      GPUContext* context, GPUSharedMemConfig shared_mem_config);

  // Launches a HIP kernel via hipLaunchKernel.
  // TODO(leary) describe the structure of kernel_params and extra in a readable
  // way.
  static bool LaunchKernel(GPUContext* context, GPUFunctionHandle function,
                           unsigned int grid_dim_x, unsigned int grid_dim_y,
                           unsigned int grid_dim_z, unsigned int block_dim_x,
                           unsigned int block_dim_y, unsigned int block_dim_z,
                           unsigned int shared_mem_bytes, GPUStreamHandle stream,
                           void** kernel_params, void** extra);

  enum class GPUBinaryType { CUDA_PTX, CUDA_CUBIN, ROCM_HSACO};
  
  // Loads gpubin_contents with the GPU Driver API and stores the resulting
  // handle in "module". Any error logs that are produced are logged internally.
  static bool LoadGPUBinary(GPUContext* context, GPUBinaryType type,
			    const char *ptx_contents, GPUModuleHandle *module);
  
  // Retrieves a named kernel from a loaded module, and places the resulting
  // handle into function (outparam) on success. Neither kernel_name nor
  // function may be null. No ownership is taken of kernel_name.
  static bool GetModuleFunction(GPUContext* context, GPUModuleHandle module,
                                const char* kernel_name,
                                GPUFunctionHandle* function);

  // Retrieves a named global/constant symbol from a loaded module, and returns
  // a device pointer and size of the symbol on success. symbol_name may not be
  // null. At least one of dptr or bytes should not be null. No ownership is
  // taken of symbol_name.
  static bool GetModuleSymbol(GPUContext* context, GPUModuleHandle module,
                              const char* symbol_name, GPUDevicePointer* dptr,
                              size_t* bytes);

  // Unloads module from the current context via cuModuleUnload.
  // TODO(leary) the documentation doesn't say what kind of disasters happen
  // if you try to unload a module while its GPUFunctionHandles are in use.
  static void UnloadModule(GPUContext* context, GPUModuleHandle module);

  // Performs a synchronous memset of the device memory segment via hipMemsetD8.
  static bool SynchronousMemsetUint8(GPUContext* context,
                                     GPUDevicePointer location, uint8 value,
                                     size_t size);

  // Performs a synchronous memset of the device memory segment via hipMemsetD32.
  static bool SynchronousMemsetUint32(GPUContext* context,
                                      GPUDevicePointer location, uint32 value,
                                      size_t uint32_count);

  // Performs an asynchronous memset of the device memory segment via
  // hipMemsetD8Async.
  static bool AsynchronousMemsetUint8(GPUContext* context,
                                      GPUDevicePointer location, uint8 value,
                                      size_t uint32_count, GPUStreamHandle stream);

  // Performs an asynchronous memset of the device memory segment via
  // hipMemsetD32Async.
  static bool AsynchronousMemsetUint32(GPUContext* context,
                                       GPUDevicePointer location, uint32 value,
                                       size_t uint32_count, GPUStreamHandle stream);

  // -- Synchronous memcopies.

  static port::Status SynchronousMemcpyD2H(GPUContext* context, void* host_dst,
                                           GPUDevicePointer gpu_src, uint64 size);
  static port::Status SynchronousMemcpyH2D(GPUContext* context,
                                           GPUDevicePointer gpu_dst,
                                           const void* host_src, uint64 size);
  static port::Status SynchronousMemcpyD2D(GPUContext* context,
                                           GPUDevicePointer gpu_dst,
                                           GPUDevicePointer gpu_src, uint64 size);

  // -- Asynchronous memcopies.

  static bool AsynchronousMemcpyD2H(GPUContext* context, void* host_dst,
                                    GPUDevicePointer gpu_src, uint64 size,
                                    GPUStreamHandle stream);
  static bool AsynchronousMemcpyH2D(GPUContext* context,
                                    GPUDevicePointer gpu_dst,
                                    const void* host_src, uint64 size,
                                    GPUStreamHandle stream);
  static bool AsynchronousMemcpyD2D(GPUContext* context,
                                    GPUDevicePointer gpu_dst,
                                    GPUDevicePointer gpu_src, uint64 size,
                                    GPUStreamHandle stream);

  // The ROCM stream callback type signature.
  // The data passed to AddStreamCallback is subsequently passed to this
  // callback when it fires.
  //
  // Some notable things:
  // * Callbacks must not make any ROCM API calls.
  // * Callbacks from independent streams execute in an undefined order and may
  //   be serialized.
  typedef void (*StreamCallback)(GPUStreamHandle stream, GPUStatus status, void *data);

  // Enqueues a callback operation into stream.
  // See StreamCallback above ROCM documentation for additional
  // details.
  static bool AddStreamCallback(GPUContext* context, GPUStreamHandle stream,
                                StreamCallback callback, void* data);

  // Causes stream to wait for event to trigger before proceeding via
  // hipStreamWaitEvent.
  static bool WaitStreamOnEvent(GPUContext* context, GPUStreamHandle stream,
                                GPUEventHandle event);

  // Blocks the calling thread until the operations enqueued onto stream have
  // been completed, via hipStreamSynchronize.
  //
  // TODO(leary) if a pathological thread enqueues operations onto the stream
  // while another thread blocks like this, can you wind up waiting an unbounded
  // amount of time?
  //
  static port::Status SynchronizeStream(GPUContext* context,
                                        GPUStreamHandle stream);

  // Blocks the calling thread until the operations associated with the context
  // have been completed, via hipCtxSynchronize.
  static bool SynchronizeContext(GPUContext* context);

  // Returns true if all stream tasks have completed at time of the call. Note
  // the potential for races around this call (if another thread adds work to
  // the stream immediately after this returns).
  static bool IsStreamIdle(GPUContext* context, GPUStreamHandle stream);

  // Returns whether code in the from context can access memory in the to
  // context via hipDeviceCanAccessPeer.
  static bool CanEnablePeerAccess(GPUContext* from, GPUContext* to);

  // Enables peer access per CanEnablePeerAccess, via hipDeviceEnablePeerAccess.
  static port::Status EnablePeerAccess(GPUContext* from, GPUContext* to);

  // Returns the elapsed milliseconds between start and stop via
  // hipEventElapsedTime.
  static bool GetEventElapsedTime(GPUContext* context,
                                  float* elapsed_milliseconds, GPUEventHandle start,
                                  GPUEventHandle stop);

  // Records that an event occurred when execution reaches the current point in
  // thestream via hipEventRecord.
  static port::Status RecordEvent(GPUContext* context, GPUEventHandle event,
                                  GPUStreamHandle stream);

  // Polls (without blocking) to determine the status of an event - pending or
  // complete (or an error status).
  static port::StatusOr<GPUStatus> QueryEvent(GPUContext* context,
                                               GPUEventHandle event);

  // -- Device-specific calls.

  // Returns the compute capability for the device; i.e (3, 5).
  // This is currently done via the deprecated device API.
  // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1ge2091bbac7e1fb18c2821612115607ea
  // (supportedd for Nvidia GPUs only)
  static port::Status GetComputeCapability(int *cc_major, int *cc_minor,
                                           GPUDeviceHandle device);

  // Returns GPU ISA version for the device; i.e 803, 900.
  // (supportedd for AMD GPUs only)
  static port::Status GetGPUISAVersion(int *version,
                                          GPUDeviceHandle device);

  // Returns the number of multiprocessors on the device (note that the device
  // may be multi-GPU-per-board).
  static port::StatusOr<int> GetMultiprocessorCount(GPUDeviceHandle device);

  // Returns the limit on number of threads that can be resident in a single
  // multiprocessor.
  static port::StatusOr<int64> GetMaxThreadsPerMultiprocessor(GPUDeviceHandle device);

  // Returns the limit on number of threads which may be resident for a single
  // block (cooperative thread array).
  static port::StatusOr<int64> GetMaxThreadsPerBlock(GPUDeviceHandle device);

  // Returns the amount of shared memory available on a single GPU core (i.e.
  // CU on ROCM devices).
  static port::StatusOr<int64> GetMaxSharedMemoryPerCore(GPUDeviceHandle device);

  // Returns the amount of shared memory available for a single block
  // (cooperative thread array).
  static port::StatusOr<int64> GetMaxSharedMemoryPerBlock(GPUDeviceHandle device);

  // Returns the maximum supported number of registers per block.
  static port::StatusOr<int64> GetMaxRegistersPerBlock(GPUDeviceHandle device);

  // Returns the number of threads per warp.
  static port::StatusOr<int64> GetThreadsPerWarp(GPUDeviceHandle device);

  // Queries the grid limits for device with hipDeviceGetAttribute calls.
  static bool GetGridLimits(int *x, int *y, int *z, GPUDeviceHandle device);

  // Returns a grab-bag of device properties in a caller-owned device_properties
  // structure for device_ordinal via hipDeviceGetProperties.
  static bool GetDeviceProperties(GPUDeviceProperty *device_properties,
                                  int device_ordinal);

  // Gets a specific integer-valued property about the given device.
  static port::StatusOr<int> GetDeviceAttribute(GPUDeviceAttribute attribute,
                                                GPUDeviceHandle device);

  // Returns whether ECC is enabled for the given GPUDeviceHandle via
  // hipDeviceGetattribute with CU_DEVICE_ATTRIBUTE_ECC_ENABLED.
  static bool IsEccEnabled(GPUDeviceHandle device, bool *result);

  // Returns the total amount of memory available for allocation by the ROCM
  // context, in bytes, via hipDeviceTotalMem.
  static bool GetDeviceTotalMemory(GPUDeviceHandle device, uint64 *result);

  // Returns the free amount of memory and total amount of memory, as reported
  // by hipMemGetInfo.
  static bool GetDeviceMemoryInfo(GPUContext* context, int64* free,
                                  int64* total);

  // Returns a PCI bus id string for the device.
  // [domain]:[bus]:[device].[function]
  static string GetPCIBusID(GPUDeviceHandle device);

  // -- Context- and device-independent calls.

  // Returns the number of visible ROCM device via hipDeviceGetCount.
  // This should correspond to the set of device ordinals available.
  static int GetDeviceCount();

  // Returns the driver version number via cuDriverGetVersion.
  // This is, surprisingly, NOT the actual driver version (e.g. 331.79) but,
  // instead, the ROCM toolkit release number that this driver is compatible
  // with; e.g. 6000 (for a ROCM 6.0 compatible driver) or 6050 (for a ROCM 6.5
  // compatible driver).
  //
  static bool GetDriverVersion(int *driver_version);

  // -- Other calls

  // Returns the maximum number of blocks (per multiprocessor) occupied by the
  // specified kernel/GPUFunctionHandle when launched with the specified parameters.
  static port::StatusOr<int> GetMaxOccupiedBlocksPerCore(
      GPUContext* context, GPUFunctionHandle kernel, int threads_per_block,
      size_t dynamic_shared_memory_bytes);

  // Seam for injecting an error at CUDA initialization time for testing
  // purposes.
  static bool driver_inject_init_error_;
};

// Ensures the given context is activated within a scope.
class ScopedActivateContext {
 public:
  // Activates the given device , if it is not the currently active device
  explicit ScopedActivateContext(GPUContext* context);

  // Checks that the device has remained activated for the duration of the
  // scope.
  ~ScopedActivateContext();

 private:
  GPUContext* to_restore_ = nullptr;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_GPU_GPU_DRIVER_H_
