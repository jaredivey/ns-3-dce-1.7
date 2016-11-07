#ifndef LIBC_H
#define LIBC_H

#include <stdarg.h>
#define _SYS_SELECT_H
#include <sys/types.h>
#undef _SYS_SELECT_H
#include </usr/local/cuda-7.5/include/cuda_runtime_api.h>
#define _STDLIB_H
#include </usr/local/cuda-7.5/include/cuda.h>
#undef _STDLIB_H
#include </usr/local/cuda-7.5/include/host_defines.h>
#include </usr/local/cuda-7.5/include/builtin_types.h>
#include </usr/local/cuda-7.5/include/driver_types.h>

struct Libc
{

#define DCE(name) void (*name ## _fn) (...);

#define DCET(rtype, name) DCE (name)
#define NATIVET(rtype, name) NATIVE (name)

#define DCE_EXPLICIT(name,rtype,...) rtype (*name ## _fn) (__VA_ARGS__);
#include "libc-ns3.h"

  char* (*strpbrk_fn)(const char *s, const char *accept);
  char* (*strstr_fn)(const char *a, const char *b);
  int (*vsnprintf_fn)(char *str, size_t size, const char *format, va_list v);

  // cudadr-context.cc
  CUresult (*cuCtxCreate_fn)(CUcontext *pctx, unsigned int flags, CUdevice dev);
  CUresult (*cuCtxAttach_fn)(CUcontext *pctx, unsigned int flags);
  CUresult (*cuCtxDestroy_fn)(CUcontext ctx);
  CUresult (*cuCtxDetach_fn)(CUcontext ctx);
  CUresult (*cuCtxGetDevice_fn)(CUdevice *device);
  CUresult (*cuCtxPopCurrent_fn)(CUcontext *pctx);
  CUresult (*cuCtxPushCurrent_fn)(CUcontext ctx);
  CUresult (*cuCtxSynchronize_fn)(void);

  // cudadr-device.cc
  CUresult (*cuDeviceComputeCapability_fn)(int *major, int *minor, CUdevice dev);
  CUresult (*cuDeviceGet_fn)(CUdevice *device, int ordinal);
  CUresult (*cuDeviceGetAttribute_fn)(int *pi, CUdevice_attribute attrib, CUdevice dev);
  CUresult (*cuDeviceGetCount_fn)(int *count);
  CUresult (*cuDeviceGetName_fn)(char *name, int len, CUdevice dev);
  CUresult (*cuDeviceGetProperties_fn)(CUdevprop *prop, CUdevice dev);
  CUresult (*cuDeviceTotalMem_fn)(size_t *bytes, CUdevice dev);

  // cudadr-event.cc
  CUresult (*cuEventCreate_fn)(CUevent *phEvent, unsigned int Flags);
  CUresult (*cuEventDestroy_fn)(CUevent hEvent);
  CUresult (*cuEventElapsedTime_fn)(float *pMilliseconds, CUevent hStart, CUevent hEnd);
  CUresult (*cuEventQuery_fn)(CUevent hEvent);
  CUresult (*cuEventRecord_fn)(CUevent hEvent, CUstream hStream);
  CUresult (*cuEventSynchronize_fn)(CUevent hEvent);

  // cudadr-execution.cc
  CUresult (*cuParamSetSize_fn)(CUfunction hfunc, unsigned int numbytes);
  CUresult (*cuFuncSetBlockShape_fn)(CUfunction hfunc, int x, int y, int z);
  CUresult (*cuLaunchGrid_fn)(CUfunction f, int grid_width, int grid_height);
  CUresult (*cuFuncGetAttribute_fn)(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
  CUresult (*cuFuncSetSharedSize_fn)(CUfunction hfunc, unsigned int bytes);
  CUresult (*cuLaunch_fn)(CUfunction f);
  CUresult (*cuLaunchKernel_fn)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
			unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
			unsigned int sharedMemBytes, CUstream hStream, void ** kernelParams, void ** extra);
  CUresult (*cuLaunchGridAsync_fn)(CUfunction f, int grid_width, int grid_height, CUstream hStream);
  CUresult (*cuParamSetf_fn)(CUfunction hfunc, int offset, float value);
  CUresult (*cuParamSeti_fn)(CUfunction hfunc, int offset, unsigned int value);
  CUresult (*cuParamSetTexRef_fn)(CUfunction hfunc, int texunit, CUtexref hTexRef);
  CUresult (*cuParamSetv_fn)(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
  CUresult (*cuFuncSetCacheConfig_fn)(CUfunction hfunc, CUfunc_cache config);

  // cudadr-initialization.cc
  CUresult (*cuInit_fn)(unsigned int flags);

  // cudadr-memory.cc
  CUresult (*cuMemFree_fn)(CUdeviceptr dptr);
  CUresult (*cuMemAlloc_fn)(CUdeviceptr *dptr, size_t bytesize);
  CUresult (*cuMemcpyDtoH_fn)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
  CUresult (*cuArrayCreate_fn)(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
  CUresult (*cuArray3DCreate_fn)(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
  CUresult (*cuMemcpy2D_fn)(const CUDA_MEMCPY2D *pCopy);
  CUresult (*cuArrayDestroy_fn)(CUarray hArray);
  CUresult (*cuMemAllocPitch_fn)(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
  CUresult (*cuMemGetAddressRange_fn)(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
  CUresult (*cuMemGetInfo_fn)(size_t *free, size_t *total);

  // cudadr-module.cc
  CUresult (*cuModuleLoadData_fn)(CUmodule *module, const void *image);
  CUresult (*cuModuleGetFunction_fn)(CUfunction *hfunc, CUmodule hmod, const char *name);
  CUresult (*cuModuleGetGlobal_fn)(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
  CUresult (*cuModuleGetTexRef_fn)(CUtexref *pTexRef, CUmodule hmod, const char *name);
  CUresult (*cuModuleLoadDataEx_fn)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
  CUresult (*cuModuleUnload_fn)(CUmodule hmod);

  // cudadr-stream.cc
  CUresult (*cuStreamCreate_fn)(CUstream *phStream, unsigned int Flags);
  CUresult (*cuStreamDestroy_fn)(CUstream hStream);
  CUresult (*cuStreamQuery_fn)(CUstream hStream);
  CUresult (*cuStreamSynchronize_fn)(CUstream hStream);

  // cudadr-texture.cc
  CUresult (*cuTexRefSetArray_fn)(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
  CUresult (*cuTexRefSetAddressMode_fn)(CUtexref hTexRef, int dim, CUaddress_mode am);
  CUresult (*cuTexRefSetFilterMode_fn)(CUtexref hTexRef, CUfilter_mode fm);
  CUresult (*cuTexRefSetFlags_fn)(CUtexref hTexRef, unsigned int Flags);
  CUresult (*cuTexRefSetFormat_fn)(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
  CUresult (*cuTexRefGetAddress_fn)(CUdeviceptr *pdptr, CUtexref hTexRef);
  CUresult (*cuTexRefGetArray_fn)(CUarray *phArray, CUtexref hTexRef);
  CUresult (*cuTexRefGetFlags_fn)(unsigned int *pFlags, CUtexref hTexRef);
  CUresult (*cuTexRefSetAddress_fn)(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);

  // cudadr-version.cc
  CUresult (*cuDriverGetVersion_fn)(int *driverVersion);

  // cudart-memory.cc
  cudaError_t (*cudaFree_fn)(void *devPtr);
  cudaError_t (*cudaFreeArray_fn)(cudaArray *array);
  cudaError_t (*cudaFreeHost_fn)(void *ptr);
  cudaError_t (*cudaGetSymbolAddress_fn)(void **devPtr, const void *symbol);
  cudaError_t (*cudaGetSymbolSize_fn)(size_t *size, const void *symbol);
  cudaError_t (*cudaHostAlloc_fn)(void **ptr, size_t size, unsigned int flags);
  cudaError_t (*cudaHostGetDevicePointer_fn)(void **pDevice, void *pHost,
          unsigned int flags);
  cudaError_t (*cudaHostGetFlags_fn)(unsigned int *pFlags, void *pHost);
  cudaError_t (*cudaMalloc_fn)(void **devPtr, size_t size);
  cudaError_t (*cudaMalloc3D_fn)(cudaPitchedPtr *pitchedDevPtr,
          cudaExtent extent);
  #if CUDART_VERSION >= 3000
  cudaError_t (*cudaMalloc3DArray_fn)(cudaArray **arrayPtr,
          const cudaChannelFormatDesc *desc, cudaExtent extent,
          unsigned int flags);
  #else
  cudaError_t (*cudaMalloc3DArray_fn)(cudaArray **arrayPtr,
          const cudaChannelFormatDesc *desc, cudaExtent extent);
  #endif

  #if CUDART_VERSION >= 3000
  cudaError_t (*cudaMallocArray_fn)(cudaArray **arrayPtr,
          const cudaChannelFormatDesc *desc, size_t width, size_t height,
          unsigned int flags);
  #else
  cudaError_t (*cudaMallocArray_fn)(cudaArray **arrayPtr,
          const cudaChannelFormatDesc *desc, size_t width, size_t height);
  #endif

  cudaError_t (*cudaMallocHost_fn)(void **ptr, size_t size);
  cudaError_t (*cudaMallocPitch_fn)(void **devPtr,
          size_t *pitch, size_t width, size_t height);
  cudaError_t (*cudaMemcpy_fn)(void *dst, const void *src, size_t count,
          cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpy2D_fn)(void *dst, size_t dpitch, const void *src,
          size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpy2DArrayToArray_fn)(cudaArray *dst, size_t wOffsetDst,
          size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
          size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpy2DAsync_fn)(void *dst, size_t dpitch, const void *src,
          size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
          cudaStream_t stream);
  cudaError_t (*cudaMemcpy2DFromArray_fn)(void *dst, size_t dpitch,
          const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
          size_t height, cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpy2DFromArrayAsync_fn)(void *dst, size_t dpitch,
          const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
          size_t height, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t (*cudaMemcpy2DToArray_fn)(cudaArray *dst, size_t wOffset,
          size_t hOffset, const void *src, size_t spitch, size_t width,
          size_t height, cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpy2DToArrayAsync_fn)(cudaArray *dst, size_t wOffset,
          size_t hOffset, const void *src, size_t spitch, size_t width,
          size_t height, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t (*cudaMemcpy3D_fn)(const cudaMemcpy3DParms *p);
  cudaError_t (*cudaMemcpy3DAsync_fn)(const cudaMemcpy3DParms *p,
          cudaStream_t stream);
  cudaError_t (*cudaMemcpyArrayToArray_fn)(cudaArray *dst, size_t wOffsetDst,
          size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
          size_t hOffsetSrc, size_t count,
          cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpyAsync_fn)(void *dst, const void *src, size_t count,
          cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t (*cudaMemcpyFromArray_fn)(void *dst, const cudaArray *src,
          size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpyFromArrayAsync_fn)(void *dst, const cudaArray *src,
          size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
          cudaStream_t stream);
  cudaError_t (*cudaMemcpyFromSymbol_fn)(void *dst, const void *symbol,
          size_t count, size_t offset,
          cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpyFromSymbolAsync_fn)(void *dst, const void *symbol,
          size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t (*cudaMemcpyToArray_fn)(cudaArray *dst, size_t wOffset,
          size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpyToArrayAsync_fn)(cudaArray *dst, size_t wOffset,
          size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
          cudaStream_t stream);
  cudaError_t (*cudaMemcpyToSymbol_fn)(const void *symbol, const void *src,
          size_t count, size_t offset,
          cudaMemcpyKind kind);
  cudaError_t (*cudaMemcpyToSymbolAsync_fn)(const void *symbol, const void *src,
          size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
  cudaError_t (*cudaMemset_fn)(void *devPtr, int c, size_t count);
  cudaError_t (*cudaMemset2D_fn)(void *mem, size_t pitch, int c, size_t width,
          size_t height);
  cudaError_t (*cudaMemset3D_fn)(cudaPitchedPtr pitchDevPtr, int value,
          cudaExtent extent);

  // cudart-device.cc
  cudaError_t (*cudaChooseDevice_fn)(int *device, const cudaDeviceProp *prop);
  cudaError_t (*cudaGetDevice_fn)(int *device);
  cudaError_t (*cudaGetDeviceCount_fn)(int *count);
  cudaError_t (*cudaGetDeviceProperties_fn)(cudaDeviceProp *prop, int device);
  cudaError_t (*cudaSetDevice_fn)(int device);
  #if CUDART_VERSION >= 3000
  cudaError_t (*cudaSetDeviceFlags_fn)(unsigned int flags);
  #else
  cudaError_t (*cudaSetDeviceFlags_fn)(int flags);
  #endif
  cudaError_t (*cudaSetValidDevices_fn)(int *device_arr, int len);
  cudaError_t (*cudaDeviceReset_fn)(void);

  // cudart-error.cc
  const char* (*cudaGetErrorString_fn)(cudaError_t error);
  cudaError_t (*cudaGetLastError_fn)(void);

  // cudart-event.cc
  cudaError_t (*cudaEventCreate_fn)(cudaEvent_t *event);
  #if CUDART_VERSION >= 3000
  cudaError_t (*cudaEventCreateWithFlags_fn)(cudaEvent_t *event,
          unsigned int flags);
  #else
  cudaError_t (*cudaEventCreateWithFlags_fn)(cudaEvent_t *event, int flags);
  #endif
  cudaError_t (*cudaEventDestroy_fn)(cudaEvent_t event);
  cudaError_t (*cudaEventElapsedTime_fn)(float *ms, cudaEvent_t start, cudaEvent_t end);
  cudaError_t (*cudaEventQuery_fn)(cudaEvent_t event);
  cudaError_t (*cudaEventRecord_fn)(cudaEvent_t event, cudaStream_t stream);
  cudaError_t (*cudaEventSynchronize_fn)(cudaEvent_t event);

  // cudart-execution.cc
  cudaError_t (*cudaConfigureCall_fn)(dim3 gridDim, dim3 blockDim,
          size_t sharedMem, cudaStream_t stream);
  #ifndef CUDART_VERSION
  #error CUDART_VERSION not defined
  #endif
  #if CUDART_VERSION >= 2030
  cudaError_t (*cudaFuncGetAttributes_fn)(struct cudaFuncAttributes *attr,
          const void *func);
  #endif
  cudaError_t (*cudaLaunch_fn)(const void *entry);
  cudaError_t (*cudaSetDoubleForDevice_fn)(double *d);
  cudaError_t (*cudaSetDoubleForHost_fn)(double *d);
  cudaError_t (*cudaSetupArgument_fn)(const void *arg, size_t size,
          size_t offset);

  // cudart-internal.cc
  void** (*__cudaRegisterFatBinary_fn)(void *fatCubin);
  void (*__cudaUnregisterFatBinary_fn)(void **fatCubinHandle);
  void (*__cudaRegisterFunction_fn)(void **fatCubinHandle, const char *hostFun,
          char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
          uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
  void (*__cudaRegisterVar_fn)(void **fatCubinHandle, char *hostVar,
          char *deviceAddress, const char *deviceName, int ext, int size,
          int constant, int global);
  void (*__cudaRegisterShared_fn)(void **fatCubinHandle, void **devicePtr);
  void (*__cudaRegisterSharedVar_fn)(void **fatCubinHandle, void **devicePtr,
          size_t size, size_t alignment, int storage);
  void (*__cudaRegisterTexture_fn)(void **fatCubinHandle,
          const textureReference *hostVar, void **deviceAddress, char *deviceName,
          int dim, int norm, int ext);
  int (*__cudaSynchronizeThreads_fn)(void** x, void* y);
  void (*__cudaTextureFetch_fn)(const void *tex, void *index, int integer,
          void *val);

  // cudart-stream.cc
  cudaError_t (*cudaStreamCreate_fn)(cudaStream_t *pStream);
  cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t stream);
  cudaError_t (*cudaStreamQuery_fn)(cudaStream_t stream);
  cudaError_t (*cudaStreamSynchronize_fn)(cudaStream_t stream);

  // cudart-texture.cc
  cudaError_t (*cudaBindTexture_fn)(size_t *offset,
          const textureReference *texref, const void *devPtr,
          const cudaChannelFormatDesc *desc, size_t size);
  cudaError_t (*cudaBindTexture2D_fn)(size_t *offset,
          const textureReference *texref, const void *devPtr,
          const cudaChannelFormatDesc *desc, size_t width, size_t height,
          size_t pitch);
  cudaError_t (*cudaBindTextureToArray_fn)(const textureReference *texref,
          const cudaArray *array, const cudaChannelFormatDesc *desc);
  cudaChannelFormatDesc (*cudaCreateChannelDesc_fn)(int x, int y, int z, int w,
          cudaChannelFormatKind f);
  cudaError_t (*cudaGetChannelDesc_fn)(cudaChannelFormatDesc *desc,
          const cudaArray *array);
  cudaError_t (*cudaGetTextureAlignmentOffset_fn)(size_t *offset,
          const textureReference *texref);
  cudaError_t (*cudaGetTextureReference_fn)(const textureReference **texref,
          const void *symbol);
  cudaError_t (*cudaUnbindTexture_fn)(const textureReference *texref);

  // cudart-thread.cc
  cudaError_t (*cudaThreadSynchronize_fn)();
  cudaError_t (*cudaThreadExit_fn)();

  // cudart-version.cc
  cudaError_t (*cudaDriverGetVersion_fn)(int *driverVersion);
  cudaError_t (*cudaRuntimeGetVersion_fn)(int *runtimeVersion);


};


#endif /* LIBC_H */
