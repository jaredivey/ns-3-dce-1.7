
#include "libc.h"

struct Libc g_libc;

// macros stolen from glibc.
#define weak_alias(name, aliasname) \
  extern __typeof (name) aliasname __attribute__ ((weak, alias (# name)))

extern "C" {

// Step 2.  Very dirty trick to force redirection to library functions
// This will work only with GCC. Number 128 was picked to be arbitrarily large to allow
// function calls with a large number of arguments.
// \see http://tigcc.ticalc.org/doc/gnuexts.html#SEC67___builtin_apply_args
// FIXME: 120925: 128 was heuristically picked to pass the test under 32bits environment.
#define NATIVE DCE
#define NATIVET DCET
#define NATIVE_WITH_ALIAS DCE_WITH_ALIAS
#define NATIVE_WITH_ALIAS2 DCE_WITH_ALIAS2

#define GCC_BT_NUM_ARGS 128

#define GCC_BUILTIN_APPLY(export_symbol, func_to_call) \
  void export_symbol (...) { \
    void *args =  __builtin_apply_args (); \
    void *result = __builtin_apply (g_libc.func_to_call ## _fn, args, GCC_BT_NUM_ARGS); \
    __builtin_return (result); \
  }

#define GCC_BUILTIN_APPLYT(rtype, export_symbol, func_to_call) \
  rtype export_symbol (...) { \
    void *args =  __builtin_apply_args (); \
    void *result = __builtin_apply ((void (*) (...)) g_libc.func_to_call ## _fn, args, GCC_BT_NUM_ARGS); \
    __builtin_return (result); \
  }


#define DCE(name)                                                               \
  GCC_BUILTIN_APPLY (name,name)

#define DCET(rtype,name)                                                               \
  GCC_BUILTIN_APPLYT (rtype,name,name)

/* From gcc/testsuite/gcc.dg/cpp/vararg2.c */
/* C99 __VA_ARGS__ versions */
#define c99_count(...)    _c99_count1 (, ## __VA_ARGS__) /* If only ## worked.*/
#define _c99_count1(...)  _c99_count2 (__VA_ARGS__,10,9,8,7,6,5,4,3,2,1,0)
#define _c99_count2(_,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,n,...) n

#define FULL_ARGS_0()
#define FULL_ARGS_1(X0)  X0 a0
#define FULL_ARGS_2(X0,X1)  X0 a0, X1 a1
#define FULL_ARGS_3(X0,X1,X2)  X0 a0, X1 a1, X2 a2
#define FULL_ARGS_4(X0,X1,X2,X3)  X0 a0, X1 a1, X2 a2, X3 a3
#define FULL_ARGS_5(X0,X1,X2,X3,X4)  X0 a0, X1 a1, X2 a2, X3 a3, X4 a4

#define _ARGS_0()
#define _ARGS_1(X0)  a0
#define _ARGS_2(X0,X1)   a0, a1
#define _ARGS_3(X0,X1,X2)  a0, a1, a2
#define _ARGS_4(X0,X1,X2,X3)  a0, a1, a2, a3
#define _ARGS_5(X0,X1,X2,X3,X4) a0, a1, a2, a3, a4

#define CAT(a, ...) PRIMITIVE_CAT (a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define  FULL_ARGS(...) CAT (FULL_ARGS_,c99_count (__VA_ARGS__)) (__VA_ARGS__)
#define  ARGS(...) CAT (_ARGS_,c99_count (__VA_ARGS__)) (__VA_ARGS__)


#define DCE_EXPLICIT(name,rtype,...)                                    \
  rtype name (FULL_ARGS (__VA_ARGS__))    \
  {                                                             \
    return g_libc.name ## _fn (ARGS (__VA_ARGS__));              \
  }

#define DCE_WITH_ALIAS(name)                                    \
  GCC_BUILTIN_APPLY (__ ## name,name)                      \
  weak_alias (__ ## name, name);

#define DCE_WITH_ALIAS2(name, internal)                 \
  GCC_BUILTIN_APPLY (internal,name)                        \
  weak_alias (internal, name);


// Note: it looks like that the stdio.h header does
// not define putc and getc as macros if you include
// them from C++ so that we do need to define the putc
// and getc functions anyway.
#undef putc
#undef getc

#include "libc-ns3.h" // do the work

// weak_alias (strtol, __strtol_internal);
// weak_alias (wctype_l, __wctype_l);
// weak_alias (strdup, __strdup);

// void exit(int status)
// {
//   g_libc.exit_fn (status);
//   int a = 0;
//   while (1)
//     {
//       // loop forever to quiet compiler warning:
//       // warning: ‘noreturn’ function does return
//       a++;
//     }
// }

// void abort(void)
// {
//   g_libc.abort_fn ();
//   int a = 0;
//   while (1)
//     {
//       // loop forever to quiet compiler warning:
//       // warning: ‘noreturn’ function does return
//       a++;
//     }
// }

char * strpbrk (const char *s, const char *a)
{
  return g_libc.strpbrk_fn (s,a);
}

char * strstr (const char *u, const char *d)
{
  return g_libc.strstr_fn (u,d);
}

int snprintf (char *s, size_t si, const char *f, ...)
{
  va_list vl;
  va_start (vl, f);
  int r =  g_libc.vsnprintf_fn (s, si, f, vl);
  va_end (vl);

  return r;
}
int vsnprintf (char *s, size_t si, const char *f, va_list v)
{
  return g_libc.vsnprintf_fn (s, si, f, v);
}

// CUDA Driver API
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
	return g_libc.cuCtxCreate_fn(pctx, flags, dev);
}

CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags)
{
	return g_libc.cuCtxAttach_fn(pctx, flags);
}

CUresult cuCtxDestroy(CUcontext ctx)
{
	return g_libc.cuCtxDestroy_fn(ctx);
}

CUresult cuCtxDetach(CUcontext ctx)
{
	return g_libc.cuCtxDetach_fn(ctx);
}

CUresult cuCtxGetDevice(CUdevice *device)
{
	return g_libc.cuCtxGetDevice_fn(device);
}

CUresult cuCtxPopCurrent(CUcontext *pctx)
{
	return g_libc.cuCtxPopCurrent_fn(pctx);
}

CUresult cuCtxPushCurrent(CUcontext ctx)
{
	return g_libc.cuCtxPushCurrent_fn(ctx);
}

CUresult cuCtxSynchronize(void)
{
	return g_libc.cuCtxSynchronize_fn();
}

CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
	return g_libc.cuDeviceComputeCapability_fn(major, minor, dev);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
	return g_libc.cuDeviceGet_fn (device, ordinal);
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
	return g_libc.cuDeviceGetAttribute_fn(pi, attrib, dev);
}

CUresult cuDeviceGetCount(int *count)
{
	return g_libc.cuDeviceGetCount_fn(count);
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
	return g_libc.cuDeviceGetName_fn(name, len, dev);
}

CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
{
	return g_libc.cuDeviceGetProperties_fn(prop, dev);
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
	return g_libc.cuDeviceTotalMem_fn(bytes, dev);
}

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
	return g_libc.cuEventCreate_fn(phEvent, Flags);
}

CUresult cuEventDestroy(CUevent hEvent)
{
	return g_libc.cuEventDestroy_fn(hEvent);
}

CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
	return g_libc.cuEventElapsedTime_fn(pMilliseconds, hStart, hEnd);
}

CUresult cuEventQuery(CUevent hEvent)
{
	return g_libc.cuEventQuery_fn(hEvent);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream)
{
	return g_libc.cuEventRecord_fn(hEvent, hStream);
}

CUresult cuEventSynchronize(CUevent hEvent)
{
	return g_libc.cuEventSynchronize_fn(hEvent);
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes)
{
	return g_libc.cuParamSetSize_fn(hfunc, numbytes);
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z)
{
	return g_libc.cuFuncSetBlockShape_fn(hfunc, x, y, z);
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height)
{
	return g_libc.cuLaunchGrid_fn(f, grid_width, grid_height);
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
{
	return g_libc.cuFuncGetAttribute_fn(pi, attrib, hfunc);
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes)
{
	return g_libc.cuFuncSetSharedSize_fn(hfunc, bytes);
}

CUresult cuLaunch(CUfunction f)
{
	return g_libc.cuLaunch_fn(f);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
		unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
		unsigned int sharedMemBytes, CUstream hStream, void ** kernelParams, void ** extra)
{
	return g_libc.cuLaunchKernel_fn(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
			sharedMemBytes, hStream, kernelParams, extra);
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream)
{
	return g_libc.cuLaunchGridAsync_fn(f, grid_width, grid_height, hStream);
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value)
{
	return g_libc.cuParamSetf_fn(hfunc, offset, value);
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value)
{
	return g_libc.cuParamSeti_fn(hfunc, offset, value);
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef)
{
	return g_libc.cuParamSetTexRef_fn(hfunc, texunit, hTexRef);
}

CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes)
{
	return g_libc.cuParamSetv_fn(hfunc, offset, ptr, numbytes);
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
	return g_libc.cuFuncSetCacheConfig_fn(hfunc, config);
}

CUresult cuInit(unsigned int flags)
{
	return g_libc.cuInit_fn(flags);
}

CUresult cuMemFree(CUdeviceptr dptr)
{
	return g_libc.cuMemFree_fn(dptr);
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize)
{
	return g_libc.cuMemAlloc_fn(dptr, bytesize);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
	return g_libc.cuMemcpyDtoH_fn(dstHost, srcDevice, ByteCount);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
	return g_libc.cuMemcpyHtoD_fn(dstDevice, srcHost, ByteCount);
}

CUresult cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
	return g_libc.cuArrayCreate_fn(pHandle, pAllocateArray);
}

CUresult cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
	return g_libc.cuArray3DCreate_fn(pHandle, pAllocateArray);
}

CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy)
{
	return g_libc.cuMemcpy2D_fn(pCopy);
}

CUresult cuArrayDestroy(CUarray hArray)
{
	return g_libc.cuArrayDestroy_fn(hArray);
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes)
{
	return g_libc.cuMemAllocPitch_fn(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr)
{
	return g_libc.cuMemGetAddressRange_fn(pbase, psize, dptr);
}

CUresult cuMemGetInfo(size_t *free, size_t *total)
{
	return g_libc.cuMemGetInfo_fn(free, total);
}

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
	return g_libc.cuModuleLoadData_fn(module, image);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
	return g_libc.cuModuleGetFunction_fn(hfunc, hmod, name);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
	return g_libc.cuModuleGetGlobal_fn(dptr, bytes, hmod, name);
}

CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name)
{
	return g_libc.cuModuleGetTexRef_fn(pTexRef, hmod, name);
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
	return g_libc.cuModuleLoadDataEx_fn(module, image, numOptions, options, optionValues);
}

CUresult cuModuleUnload(CUmodule hmod)
{
	return g_libc.cuModuleUnload_fn(hmod);
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
	return g_libc.cuStreamCreate_fn(phStream, Flags);
}

CUresult cuStreamDestroy(CUstream hStream)
{
	return g_libc.cuStreamDestroy_fn(hStream);
}

CUresult cuStreamQuery(CUstream hStream)
{
	return g_libc.cuStreamQuery_fn(hStream);
}

CUresult cuStreamSynchronize(CUstream hStream)
{
	return g_libc.cuStreamSynchronize_fn(hStream);
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags)
{
	return g_libc.cuTexRefSetArray_fn(hTexRef, hArray, Flags);
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am)
{
	return g_libc.cuTexRefSetAddressMode_fn(hTexRef, dim, am);
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm)
{
	return g_libc.cuTexRefSetFilterMode_fn(hTexRef, fm);
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags)
{
	return g_libc.cuTexRefSetFlags_fn(hTexRef, Flags);
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents)
{
	return g_libc.cuTexRefSetFormat_fn(hTexRef, fmt, NumPackedComponents);
}

CUresult cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef)
{
	return g_libc.cuTexRefGetAddress_fn(pdptr, hTexRef);
}

CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef)
{
	return g_libc.cuTexRefGetArray_fn(phArray, hTexRef);
}

CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef)
{
	return g_libc.cuTexRefGetFlags_fn(pFlags, hTexRef);
}

CUresult cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes)
{
	return g_libc.cuTexRefSetAddress_fn(ByteOffset, hTexRef, dptr, bytes);
}

CUresult cuDriverGetVersion(int *driverVersion)
{
	return g_libc.cuDriverGetVersion_fn(driverVersion);
}

// CUDA Runtime API
cudaError_t cudaFree (void *devPtr)
{
  return g_libc.cudaFree_fn (devPtr);
}

cudaError_t cudaFreeArray(cudaArray *array) {
  return g_libc.cudaFreeArray_fn(array);
}

cudaError_t cudaFreeHost(void *ptr) {
  return g_libc.cudaFreeHost_fn(ptr);
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  return g_libc.cudaGetSymbolAddress_fn(devPtr, symbol);
}

cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
  return g_libc.cudaGetSymbolSize_fn(size, symbol);
}

cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
  return g_libc.cudaHostAlloc_fn(ptr, size, flags);
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
  return g_libc.cudaHostGetDevicePointer_fn(pDevice, pHost, flags);
}

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    // Achtung: falling back to the simplest method because we can't map memory
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
  return g_libc.cudaHostGetFlags_fn(pFlags, pHost);
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  return g_libc.cudaMalloc_fn(devPtr, size);
}

cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
        cudaExtent extent) {
return g_libc.cudaMalloc3D_fn(pitchedDevPtr, extent);
}

#if CUDART_VERSION >= 3000
cudaError_t cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent,
        unsigned int flags) {
	return g_libc.cudaMalloc3DArray_fn(arrayPtr, desc, extent, flags);
#else
cudaError_t cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent) {
	return g_libc.cudaMalloc3DArray_fn(arrayPtr, desc, extent);
#endif
}

// FIXME: new mapping way

#if CUDART_VERSION >= 3000
cudaError_t cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
	return g_libc.cudaMallocArray_fn(arrayPtr, desc, width, height, flags);
#else
cudaError_t cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height) {
	return g_libc.cudaMallocArray_fn(arrayPtr, desc, width, height);
#endif
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
  return g_libc.cudaMallocHost_fn(ptr, size);
}

cudaError_t cudaMallocPitch(void **devPtr,
        size_t *pitch, size_t width, size_t height) {
  return g_libc.cudaMallocPitch_fn(devPtr, pitch, width, height);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
  return g_libc.cudaMemcpy_fn(dst, src, count, kind);
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
  return g_libc.cudaMemcpy2D_fn(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
  return g_libc.cudaMemcpy2DArrayToArray_fn(dst, wOffsetDst,
        hOffsetDst, src, wOffsetSrc,
        hOffsetSrc, width, height, kind);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t stream) {
  return g_libc.cudaMemcpy2DAsync_fn (dst, dpitch, src,
	        spitch, width, height, kind,
	        stream);
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind) {
  return g_libc.cudaMemcpy2DFromArray_fn(dst, dpitch,
        src, wOffset, hOffset, width,
        height, kind);
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
  return g_libc.cudaMemcpy2DFromArrayAsync_fn(dst, dpitch,
        src, wOffset, hOffset, width,
        height, kind, stream);
}

cudaError_t cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind) {
  return g_libc.cudaMemcpy2DToArray_fn(dst, wOffset,
        hOffset, src, spitch, width,
        height, kind);
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
  return g_libc.cudaMemcpy2DToArrayAsync_fn(dst, wOffset,
        hOffset, src, spitch, width,
        height, kind, stream);
}

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p) {
  return g_libc.cudaMemcpy3D_fn(p);
}

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
        cudaStream_t stream) {
  return g_libc.cudaMemcpy3DAsync_fn(p, stream);
}

cudaError_t cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t count,
        cudaMemcpyKind kind) {
  return g_libc.cudaMemcpyArrayToArray_fn(dst, wOffsetDst,
        hOffsetDst, src, wOffsetSrc,
        hOffsetSrc, count,
        kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
  return g_libc.cudaMemcpyAsync_fn(dst, src, count,
        kind, stream);
}

cudaError_t cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
  return g_libc.cudaMemcpyFromArray_fn (dst, src,
        wOffset, hOffset, count, kind);
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
  return g_libc.cudaMemcpyFromArrayAsync_fn(dst, src,
        wOffset, hOffset, count, kind,
        stream);
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
  return g_libc.cudaMemcpyFromSymbol_fn(dst, symbol,
        count, offset,
        kind);
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
  return g_libc.cudaMemcpyFromSymbolAsync_fn(dst, symbol,
	        count, offset, kind, stream);
}

cudaError_t cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind) {
  return g_libc.cudaMemcpyToArray_fn(dst, wOffset,
        hOffset, src, count, kind);
}

cudaError_t cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
  return g_libc.cudaMemcpyToArrayAsync_fn(dst, wOffset,
        hOffset, src, count, kind,
        stream);
}

cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
  return g_libc.cudaMemcpyToSymbol_fn(symbol, src,
        count, offset,
        kind);
}

cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
  return g_libc.cudaMemcpyToSymbolAsync_fn(symbol, src,
        count, offset, kind, stream);
}

cudaError_t cudaMemset(void *devPtr, int c, size_t count) {
  return g_libc.cudaMemset_fn(devPtr, c, count);
}

cudaError_t cudaMemset2D(void *mem, size_t pitch, int c, size_t width,
        size_t height) {
  return g_libc.cudaMemset2D_fn(mem, pitch, c, width, height);
}

cudaError_t cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
        cudaExtent extent) {
  return g_libc.cudaMemset3D_fn(pitchDevPtr, value, extent);
}

cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
    return g_libc.cudaChooseDevice_fn(device, prop);
}

cudaError_t cudaGetDevice(int *device) {
    return g_libc.cudaGetDevice_fn(device);
}

cudaError_t cudaGetDeviceCount(int *count) {
    return g_libc.cudaGetDeviceCount_fn(count);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {

    return g_libc.cudaGetDeviceProperties_fn(prop, device);
}

cudaError_t cudaSetDevice(int device) {
    return g_libc.cudaSetDevice_fn(device);
}

#if CUDART_VERSION >= 3000
cudaError_t cudaSetDeviceFlags(unsigned int flags) {
#else
cudaError_t cudaSetDeviceFlags(int flags) {
#endif
    return g_libc.cudaSetDeviceFlags_fn(flags);
}

cudaError_t cudaSetValidDevices(int *device_arr, int len) {
    return g_libc.cudaSetValidDevices_fn(device_arr, len);
}

cudaError_t cudaDeviceReset (void) {
	return g_libc.cudaDeviceReset_fn();
}

const char* cudaGetErrorString(cudaError_t error) {
	return g_libc.cudaGetErrorString_fn(error);
}

cudaError_t cudaGetLastError(void) {
	return g_libc.cudaGetLastError_fn();
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return g_libc.cudaEventCreate_fn(event);
}

#if CUDART_VERSION >= 3000
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags) {
#else
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
#endif
	return g_libc.cudaEventCreateWithFlags_fn(event, flags);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return g_libc.cudaEventDestroy_fn(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  return g_libc.cudaEventElapsedTime_fn(ms, start, end);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return g_libc.cudaEventQuery_fn(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return g_libc.cudaEventRecord_fn(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return g_libc.cudaEventSynchronize_fn(event);
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream) {
  return g_libc.cudaConfigureCall_fn(gridDim, blockDim,
        sharedMem, stream);
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
        const void *func) {
  return g_libc.cudaFuncGetAttributes_fn(attr,
        func);
}
#endif

cudaError_t cudaLaunch(const void *entry) {
  return g_libc.cudaLaunch_fn(entry);
}

cudaError_t cudaSetDoubleForDevice(double *d) {
  return g_libc.cudaSetDoubleForDevice_fn(d);
}

cudaError_t cudaSetDoubleForHost(double *d) {
	  return g_libc.cudaSetDoubleForHost_fn(d);
}

cudaError_t cudaSetupArgument(const void *arg, size_t size,
        size_t offset) {
  return g_libc.cudaSetupArgument_fn(arg, size, offset);
}
void** __cudaRegisterFatBinary(void *fatCubin) {
  return g_libc.__cudaRegisterFatBinary_fn(fatCubin);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  g_libc.__cudaUnregisterFatBinary_fn(fatCubinHandle);
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
  g_libc.__cudaRegisterFunction_fn(fatCubinHandle, hostFun,
	        deviceFun, deviceName, thread_limit, tid,
	        bid, bDim, gDim, wSize);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) {
  g_libc.__cudaRegisterVar_fn(fatCubinHandle, hostVar,
	        deviceAddress, deviceName, ext, size,
	        constant, global);
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
  g_libc.__cudaRegisterShared_fn(fatCubinHandle, devicePtr);
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
        size_t size, size_t alignment, int storage) {
  g_libc.__cudaRegisterSharedVar_fn(fatCubinHandle, devicePtr,
	        size, alignment, storage);
}

void __cudaRegisterTexture(void **fatCubinHandle,
        const textureReference *hostVar, void **deviceAddress, char *deviceName,
        int dim, int norm, int ext) {
  g_libc.__cudaRegisterTexture_fn(fatCubinHandle,
	        hostVar, deviceAddress, deviceName,
	        dim, norm, ext);
}

int __cudaSynchronizeThreads(void** x, void* y) {
  return g_libc.__cudaSynchronizeThreads_fn(x,y);
}

void __cudaTextureFetch(const void *tex, void *index, int integer,
        void *val) {
  g_libc.__cudaTextureFetch_fn(tex, index, integer, val);
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  return g_libc.cudaStreamCreate_fn(pStream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return g_libc.cudaStreamDestroy_fn(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return g_libc.cudaStreamQuery_fn(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return g_libc.cudaStreamSynchronize_fn(stream);
}

cudaError_t cudaBindTexture(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t size) {
  return g_libc.cudaBindTexture_fn(offset, texref, devPtr, desc, size);
}

cudaError_t cudaBindTexture2D(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch) {
  return g_libc.cudaBindTexture2D_fn(offset, texref, devPtr, desc,
		  width, height, pitch);
}

cudaError_t cudaBindTextureToArray(const textureReference *texref,
        const cudaArray *array, const cudaChannelFormatDesc *desc) {
  return g_libc.cudaBindTextureToArray_fn(texref, array, desc);
}

cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
        cudaChannelFormatKind f) {
  return g_libc.cudaCreateChannelDesc_fn(x,y,z,w,f);
}

cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *desc,
        const cudaArray *array) {
  return g_libc.cudaGetChannelDesc_fn(desc, array);
}

cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
        const textureReference *texref) {
  return g_libc.cudaGetTextureAlignmentOffset_fn(offset, texref);
}

cudaError_t cudaGetTextureReference(const textureReference **texref,
        const void *symbol) {
  return g_libc.cudaGetTextureReference_fn(texref, symbol);
}

cudaError_t cudaUnbindTexture(const textureReference *texref) {
  return g_libc.cudaUnbindTexture_fn(texref);
}

cudaError_t cudaThreadSynchronize() {
  return g_libc.cudaThreadSynchronize_fn ();
}

cudaError_t cudaThreadExit() {
  return g_libc.cudaThreadExit_fn ();
}
cudaError_t cudaDriverGetVersion(int *driverVersion) {
  return g_libc.cudaDriverGetVersion_fn(driverVersion);
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
  return g_libc.cudaRuntimeGetVersion_fn(runtimeVersion);
}

#include "libc-globals.h"

void LIBSETUP (const struct Libc *fn)
{
  /* The following assignment of fn to g_libc is a bit weird: we perform a copy of the data
   * structures by hand rather than executing g_libc = fn. This is obviously done on purpose.
   * The reason is that g_libc = fn would trigger a call to the memcpy function because the
   * Libc structure is very big. The memcpy function is resolved through the dynamic loader's
   * symbol lookup mechanism to the local memcpy function and that local memcpy function happens
   * to be calling g_libc.memcpy_fn which is set to NULL before the data structure is initialized.
   */
  const unsigned char *src = (const unsigned char *)fn;
  unsigned char *dst = (unsigned char *)&g_libc;
  unsigned int i;
  for (i = 0; i < sizeof (struct Libc); ++i)
    {
      *dst = *src;
      src++;
      dst++;
    }

  setup_global_variables ();
}

#ifdef HAVE_GETCPUFEATURES
// Below there is an exception implementation: because the libm of glibc2.14 call  __get_cpu_features during dlopen,
// and during dlopen of libm  DCE do not have yet called lib_setup so there we implement __get_cpu_features
// directly without using the global g_libc variable, we can do it only if our implementation of the method
// do not interract with any ressouces of DCE or NS3 and do no call any other libc methods ...
struct cpu_features;
extern const struct cpu_features * dce___get_cpu_features (void);
const struct cpu_features * __get_cpu_features (void)
{
  return dce___get_cpu_features ();
}
#endif
} // extern "C"
