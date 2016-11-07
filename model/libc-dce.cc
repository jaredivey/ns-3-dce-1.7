#define _GNU_SOURCE 1
#undef __OPTIMIZE__
#define _LARGEFILE64_SOURCE 1

#include "libc-dce.h"
#include "libc.h"

#include "arpa/dce-inet.h"
#include "sys/dce-socket.h"
#include "sys/dce-time.h"
#include "sys/dce-ioctl.h"
#include "sys/dce-mman.h"
#include "sys/dce-stat.h"
#include "sys/dce-select.h"
#include "sys/dce-timerfd.h"
#include "sys/dce-epoll.h"
#include "dce-unistd.h"
#include "dce-netdb.h"
#include "dce-pthread.h"
#include "dce-stdio.h"
#include "dce-stdarg.h"
#include "dce-errno.h"
#include "dce-libc-private.h"
#include "dce-fcntl.h"
#include "dce-sched.h"
#include "dce-poll.h"
#include "dce-signal.h"
#include "dce-stdlib.h"
#include "dce-time.h"
#include "dce-semaphore.h"
#include "dce-cxa.h"
#include "dce-string.h"
#include "dce-global-variables.h"
#include "dce-random.h"
#include "dce-umask.h"
#include "dce-misc.h"
#include "dce-wait.h"
#include "dce-locale.h"
#include "net/dce-if.h"
#include "dce-syslog.h"
#include "dce-pwd.h"
#include "dce-dirent.h"
#include "dce-vfs.h"
#include "dce-termio.h"
#include "dce-dl.h"
#include "cuda-frontend/cudart.h"
#include "cuda-frontend/cudadr.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <fcntl.h>
#include <getopt.h>
#include <grp.h>
#include <ifaddrs.h>
#include <sys/uio.h>
#include <libgen.h>
#include <locale.h>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <sys/dir.h>
#include <sys/ioctl.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <sys/timerfd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <pthread.h>
#include <pwd.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>
#include <wctype.h>
#include <xlocale.h>
#include <errno.h>
#include <setjmp.h>
#include <libintl.h>
#include <pwd.h>
#include <inttypes.h>
#include <error.h>
#include <netinet/ether.h>
#include <search.h>
#include <fnmatch.h>
#include <langinfo.h>
#include <sys/vfs.h>
#include <termio.h>
#include <math.h>
#include <dlfcn.h>
#include <assert.h>
#include <link.h>
#include <sys/syscall.h>
#include <iconv.h>
#include <sys/file.h>
#include <readline/readline.h>
#include <fenv.h>
#include <sys/eventfd.h>
#include <sched.h>

extern void __cxa_finalize (void *d);
extern int __cxa_atexit (void (*func)(void *), void *arg, void *d);

extern int (*__gxx_personality_v0)(int a, int b,
                                   unsigned c,
                                   struct _Unwind_Exception *d,
                                   struct _Unwind_Context *e);

// extern int __gxx_personality_v0 (int a, int b,
//                                                               unsigned c, struct _Unwind_Exception *d, struct _Unwind_Context *e);
// extern int __xpg_strerror_r (int __errnum, char *__buf, size_t __buflen);
extern int __xpg_strerror_r (int __errnum, char *__buf, size_t __buflen);

// from glibc's string.h
extern char * __strcpy_chk (char *__restrict __dest,
                            const char *__restrict __src,
                            size_t __destlen);

extern void * __rawmemchr (const void *s, int c);
extern void * __memcpy_chk(void * dest, const void * src, size_t len, size_t destlen);
extern char * __strncpy_chk(char * s1, const char * s2, size_t n, size_t s1len);

// from glibc's stdio.h
extern int __sprintf_chk (char *, int, size_t, const char *, ...) __THROW;
extern int __snprintf_chk (char *, size_t, int, size_t, const char *, ...)
__THROW;
extern int __vsprintf_chk (char *, int, size_t, const char *,
                           _G_va_list) __THROW;
extern int __vsnprintf_chk (char *, size_t, int, size_t, const char *,
                            _G_va_list) __THROW;
extern int __printf_chk (int, const char *, ...);
extern int __fprintf_chk (FILE *, int, const char *, ...);
extern int __vprintf_chk (int, const char *, _G_va_list);
extern int __vfprintf_chk (FILE *, int, const char *, _G_va_list);
extern char * __fgets_unlocked_chk (char *buf, size_t size, int n, FILE *fp);
extern char * __fgets_chk (char *buf, size_t size, int n, FILE *fp);
extern int __asprintf_chk (char **, int, const char *, ...) __THROW;
extern int __vasprintf_chk (char **, int, const char *, _G_va_list) __THROW;
extern int __dprintf_chk (int, int, const char *, ...);
extern int __vdprintf_chk (int, int, const char *, _G_va_list);
extern int __obstack_printf_chk (struct obstack *, int, const char *, ...)
__THROW;
extern int __obstack_vprintf_chk (struct obstack *, int, const char *,
                                  _G_va_list) __THROW;
extern void __stack_chk_fail (void);

// from glibc's bits/fnctl2.h
#ifndef __USE_FILE_OFFSET64
extern int __open_2 (const char *__path, int __oflag) __nonnull ((1));
extern int __REDIRECT (__open_alias, (const char *__path, int __oflag, ...),
		       open) __nonnull ((1));
#else
extern int __REDIRECT (__open_2, (const char *__path, int __oflag),
		       __open64_2) __nonnull ((1));
extern int __REDIRECT (__open_alias, (const char *__path, int __oflag, ...),
		       open64) __nonnull ((1));
#endif

// from glibc's bits/select2.h
extern long int __fdelt_chk (long int __d);

typedef void (*func_t)(...);

extern "C" {

void libc_dce (struct Libc **libc)
{
  *libc = new Libc;

#define DCE(name) (*libc)->name ## _fn = (func_t)(__typeof (&name))dce_ ## name;
#define DCET(rtype,name) DCE (name)
#define DCE_EXPLICIT(name,rtype,...) (*libc)->name ## _fn = dce_ ## name;

#define NATIVE(name)                                                    \
  (*libc)->name ## _fn = (func_t)name;
#define NATIVET(rtype, name) NATIVE(name)

#define NATIVE_EXPLICIT(name, type)                             \
  (*libc)->name ## _fn = (func_t)((type)name);

#include "libc-ns3.h"

  (*libc)->strpbrk_fn = dce_strpbrk;
  (*libc)->strstr_fn = dce_strstr;
  (*libc)->vsnprintf_fn = dce_vsnprintf;

  // cudadr-context.cc
  (*libc)->cuCtxCreate_fn = dce_cuCtxCreate;
  (*libc)->cuCtxAttach_fn = dce_cuCtxAttach;
  (*libc)->cuCtxDestroy_fn = dce_cuCtxDestroy;
  (*libc)->cuCtxDetach_fn = dce_cuCtxDetach;
  (*libc)->cuCtxGetDevice_fn = dce_cuCtxGetDevice;
  (*libc)->cuCtxPopCurrent_fn = dce_cuCtxPopCurrent;
  (*libc)->cuCtxPushCurrent_fn = dce_cuCtxPushCurrent;
  (*libc)->cuCtxSynchronize_fn = dce_cuCtxSynchronize;

  // cudadr-device.cc
  (*libc)->cuDeviceComputeCapability_fn = dce_cuDeviceComputeCapability;
  (*libc)->cuDeviceGet_fn = dce_cuDeviceGet;
  (*libc)->cuDeviceGetAttribute_fn = dce_cuDeviceGetAttribute;
  (*libc)->cuDeviceGetCount_fn = dce_cuDeviceGetCount;
  (*libc)->cuDeviceGetName_fn = dce_cuDeviceGetName;
  (*libc)->cuDeviceGetProperties_fn = dce_cuDeviceGetProperties;
  (*libc)->cuDeviceTotalMem_fn = dce_cuDeviceTotalMem;

  // cudadr-event.cc
  (*libc)->cuEventCreate_fn = dce_cuEventCreate;
  (*libc)->cuEventDestroy_fn = dce_cuEventDestroy;
  (*libc)->cuEventElapsedTime_fn = dce_cuEventElapsedTime;
  (*libc)->cuEventQuery_fn = dce_cuEventQuery;
  (*libc)->cuEventRecord_fn = dce_cuEventRecord;
  (*libc)->cuEventSynchronize_fn = dce_cuEventSynchronize;

  // cudadr-execution.cc
  (*libc)->cuParamSetSize_fn = dce_cuParamSetSize;
  (*libc)->cuFuncSetBlockShape_fn = dce_cuFuncSetBlockShape;
  (*libc)->cuLaunchGrid_fn = dce_cuLaunchGrid;
  (*libc)->cuFuncGetAttribute_fn = dce_cuFuncGetAttribute;
  (*libc)->cuFuncSetSharedSize_fn = dce_cuFuncSetSharedSize;
  (*libc)->cuLaunch_fn = dce_cuLaunch;
  (*libc)->cuLaunchKernel_fn = dce_cuLaunchKernel;
  (*libc)->cuLaunchGridAsync_fn = dce_cuLaunchGridAsync;
  (*libc)->cuParamSetf_fn = dce_cuParamSetf;
  (*libc)->cuParamSeti_fn = dce_cuParamSeti;
  (*libc)->cuParamSetTexRef_fn = dce_cuParamSetTexRef;
  (*libc)->cuParamSetv_fn = dce_cuParamSetv;
  (*libc)->cuFuncSetCacheConfig_fn = dce_cuFuncSetCacheConfig;

  // cudadr-initialization.cc
  (*libc)->cuInit_fn = dce_cuInit;

  // cudadr-memory.cc
  (*libc)->cuMemFree_fn = dce_cuMemFree;
  (*libc)->cuMemAlloc_fn = dce_cuMemAlloc;
  (*libc)->cuMemcpyDtoH_fn = dce_cuMemcpyDtoH;
  (*libc)->cuMemcpyHtoD_fn = dce_cuMemcpyHtoD;
  (*libc)->cuArrayCreate_fn = dce_cuArrayCreate;
  (*libc)->cuArray3DCreate_fn = dce_cuArray3DCreate;
  (*libc)->cuMemcpy2D_fn = dce_cuMemcpy2D;
  (*libc)->cuArrayDestroy_fn = dce_cuArrayDestroy;
  (*libc)->cuMemAllocPitch_fn = dce_cuMemAllocPitch;
  (*libc)->cuMemGetAddressRange_fn = dce_cuMemGetAddressRange;
  (*libc)->cuMemGetInfo_fn = dce_cuMemGetInfo;

  // cudadr-module.cc
  (*libc)->cuModuleLoadData_fn = dce_cuModuleLoadData;
  (*libc)->cuModuleGetFunction_fn = dce_cuModuleGetFunction;
  (*libc)->cuModuleGetGlobal_fn = dce_cuModuleGetGlobal;
  (*libc)->cuModuleGetTexRef_fn = dce_cuModuleGetTexRef;
  (*libc)->cuModuleLoadDataEx_fn = dce_cuModuleLoadDataEx;
  (*libc)->cuModuleUnload_fn = dce_cuModuleUnload;

  // cudadr-stream.cc
  (*libc)->cuStreamCreate_fn = dce_cuStreamCreate;
  (*libc)->cuStreamDestroy_fn = dce_cuStreamDestroy;
  (*libc)->cuStreamQuery_fn = dce_cuStreamQuery;
  (*libc)->cuStreamSynchronize_fn = dce_cuStreamSynchronize;

  // cudadr-texture.cc
  (*libc)->cuTexRefSetArray_fn = dce_cuTexRefSetArray;
  (*libc)->cuTexRefSetAddressMode_fn = dce_cuTexRefSetAddressMode;
  (*libc)->cuTexRefSetFilterMode_fn = dce_cuTexRefSetFilterMode;
  (*libc)->cuTexRefSetFlags_fn = dce_cuTexRefSetFlags;
  (*libc)->cuTexRefSetFormat_fn = dce_cuTexRefSetFormat;
  (*libc)->cuTexRefGetAddress_fn = dce_cuTexRefGetAddress;
  (*libc)->cuTexRefGetArray_fn = dce_cuTexRefGetArray;
  (*libc)->cuTexRefGetFlags_fn = dce_cuTexRefGetFlags;
  (*libc)->cuTexRefSetAddress_fn = dce_cuTexRefSetAddress;

  // cudadr-version.cc
  (*libc)->cuDriverGetVersion_fn = dce_cuDriverGetVersion;

  // cudart-memory.cc
  (*libc)->cudaFree_fn = dce_cudaFree;
  (*libc)->cudaFreeArray_fn = dce_cudaFreeArray;
  (*libc)->cudaFreeHost_fn = dce_cudaFreeHost;
  (*libc)->cudaGetSymbolAddress_fn = dce_cudaGetSymbolAddress;
  (*libc)->cudaGetSymbolSize_fn = dce_cudaGetSymbolSize;
  (*libc)->cudaHostAlloc_fn = dce_cudaHostAlloc;
  (*libc)->cudaHostGetDevicePointer_fn = dce_cudaHostGetDevicePointer;
  (*libc)->cudaHostGetFlags_fn = dce_cudaHostGetFlags;
  (*libc)->cudaMalloc_fn = dce_cudaMalloc;
  (*libc)->cudaMalloc3D_fn = dce_cudaMalloc3D;
  (*libc)->cudaMalloc3DArray_fn = dce_cudaMalloc3DArray;
  (*libc)->cudaMallocArray_fn = dce_cudaMallocArray;
  (*libc)->cudaMallocHost_fn = dce_cudaMallocHost;
  (*libc)->cudaMallocPitch_fn = dce_cudaMallocPitch;
  (*libc)->cudaMemcpy_fn = dce_cudaMemcpy;
  (*libc)->cudaMemcpy2D_fn = dce_cudaMemcpy2D;
  (*libc)->cudaMemcpy2DArrayToArray_fn = dce_cudaMemcpy2DArrayToArray;
  (*libc)->cudaMemcpy2DAsync_fn = dce_cudaMemcpy2DAsync;
  (*libc)->cudaMemcpy2DFromArray_fn = dce_cudaMemcpy2DFromArray;
  (*libc)->cudaMemcpy2DFromArrayAsync_fn = dce_cudaMemcpy2DFromArrayAsync;
  (*libc)->cudaMemcpy2DToArray_fn = dce_cudaMemcpy2DToArray;
  (*libc)->cudaMemcpy2DToArrayAsync_fn = dce_cudaMemcpy2DToArrayAsync;
  (*libc)->cudaMemcpy3D_fn = dce_cudaMemcpy3D;
  (*libc)->cudaMemcpy3DAsync_fn = dce_cudaMemcpy3DAsync;
  (*libc)->cudaMemcpyArrayToArray_fn = dce_cudaMemcpyArrayToArray;
  (*libc)->cudaMemcpyAsync_fn = dce_cudaMemcpyAsync;
  (*libc)->cudaMemcpyFromArray_fn = dce_cudaMemcpyFromArray;
  (*libc)->cudaMemcpyFromArrayAsync_fn = dce_cudaMemcpyFromArrayAsync;
  (*libc)->cudaMemcpyFromSymbol_fn = dce_cudaMemcpyFromSymbol;
  (*libc)->cudaMemcpyFromSymbolAsync_fn = dce_cudaMemcpyFromSymbolAsync;
  (*libc)->cudaMemcpyToArray_fn = dce_cudaMemcpyToArray;
  (*libc)->cudaMemcpyToArrayAsync_fn = dce_cudaMemcpyToArrayAsync;
  (*libc)->cudaMemcpyToSymbol_fn = dce_cudaMemcpyToSymbol;
  (*libc)->cudaMemcpyToSymbolAsync_fn = dce_cudaMemcpyToSymbolAsync;
  (*libc)->cudaMemset_fn = dce_cudaMemset;
  (*libc)->cudaMemset2D_fn = dce_cudaMemset2D;
  (*libc)->cudaMemset3D_fn = dce_cudaMemset3D;

  // cudart-device.cc
  (*libc)->cudaChooseDevice_fn = dce_cudaChooseDevice;
  (*libc)->cudaGetDevice_fn = dce_cudaGetDevice;
  (*libc)->cudaGetDeviceCount_fn = dce_cudaGetDeviceCount;
  (*libc)->cudaGetDeviceProperties_fn = dce_cudaGetDeviceProperties;
  (*libc)->cudaSetDevice_fn = dce_cudaSetDevice;
  (*libc)->cudaSetDeviceFlags_fn = dce_cudaSetDeviceFlags;
  (*libc)->cudaSetValidDevices_fn = dce_cudaSetValidDevices;
  (*libc)->cudaDeviceReset_fn = dce_cudaDeviceReset;

  // cudart-error.cc
  (*libc)->cudaGetErrorString_fn = dce_cudaGetErrorString;
  (*libc)->cudaGetLastError_fn = dce_cudaGetLastError;

  // cudart-event.cc
  (*libc)->cudaEventCreate_fn = dce_cudaEventCreate;
  (*libc)->cudaEventCreateWithFlags_fn = dce_cudaEventCreateWithFlags;
  (*libc)->cudaEventDestroy_fn = dce_cudaEventDestroy;
  (*libc)->cudaEventElapsedTime_fn = dce_cudaEventElapsedTime;
  (*libc)->cudaEventQuery_fn = dce_cudaEventQuery;
  (*libc)->cudaEventRecord_fn = dce_cudaEventRecord;
  (*libc)->cudaEventSynchronize_fn = dce_cudaEventSynchronize;

  // cudart-execution.cc
  (*libc)->cudaConfigureCall_fn = dce_cudaConfigureCall;
  #ifndef CUDART_VERSION
  #error CUDART_VERSION not defined
  #endif
  #if CUDART_VERSION >= 2030
  (*libc)->cudaFuncGetAttributes_fn = dce_cudaFuncGetAttributes;
  #endif
  (*libc)->cudaLaunch_fn = dce_cudaLaunch;
  (*libc)->cudaSetDoubleForDevice_fn = dce_cudaSetDoubleForDevice;
  (*libc)->cudaSetDoubleForHost_fn = dce_cudaSetDoubleForHost;
  (*libc)->cudaSetupArgument_fn = dce_cudaSetupArgument;

  // cudart-internal.cc
  (*libc)->__cudaRegisterFatBinary_fn = dce___cudaRegisterFatBinary;
  (*libc)->__cudaUnregisterFatBinary_fn = dce___cudaUnregisterFatBinary;
  (*libc)->__cudaRegisterFunction_fn = dce___cudaRegisterFunction;
  (*libc)->__cudaRegisterVar_fn = dce___cudaRegisterVar;
  (*libc)->__cudaRegisterShared_fn = dce___cudaRegisterShared;
  (*libc)->__cudaRegisterSharedVar_fn = dce___cudaRegisterSharedVar;
  (*libc)->__cudaRegisterTexture_fn = dce___cudaRegisterTexture;
  (*libc)->__cudaSynchronizeThreads_fn = dce___cudaSynchronizeThreads;
  (*libc)->__cudaTextureFetch_fn = dce___cudaTextureFetch;

  // cudart-stream.cc
  (*libc)->cudaStreamCreate_fn = dce_cudaStreamCreate;
  (*libc)->cudaStreamDestroy_fn = dce_cudaStreamDestroy;
  (*libc)->cudaStreamQuery_fn = dce_cudaStreamQuery;
  (*libc)->cudaStreamSynchronize_fn = dce_cudaStreamSynchronize;

  // cudart-texture.cc
  (*libc)->cudaBindTexture_fn = dce_cudaBindTexture;
  (*libc)->cudaBindTexture2D_fn = dce_cudaBindTexture2D;
  (*libc)->cudaBindTextureToArray_fn = dce_cudaBindTextureToArray;
  (*libc)->cudaGetChannelDesc_fn = dce_cudaGetChannelDesc;
  (*libc)->cudaGetTextureAlignmentOffset_fn = dce_cudaGetTextureAlignmentOffset;
  (*libc)->cudaGetTextureReference_fn = dce_cudaGetTextureReference;
  (*libc)->cudaUnbindTexture_fn = dce_cudaUnbindTexture;

  // cudart-thread.cc
  (*libc)->cudaThreadSynchronize_fn = dce_cudaThreadSynchronize;
  (*libc)->cudaThreadExit_fn = dce_cudaThreadExit;

  // cudart-version.cc
  (*libc)->cudaDriverGetVersion_fn = dce_cudaDriverGetVersion;
  (*libc)->cudaRuntimeGetVersion_fn = dce_cudaRuntimeGetVersion;


}
} // extern "C"

