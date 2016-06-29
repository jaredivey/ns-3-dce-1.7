/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   CudaRt.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Fri Oct 9 15:55:40 2009
 *
 * @brief
 *
 *
 */

#ifndef _CUDART_H
#define	_CUDART_H

#include </usr/local/cuda/include/cuda_runtime_api.h>

#include "ns3/utils.h"
#include "ns3/cuda-util.h"
#include "cudart-frontend.h"

#ifdef __cplusplus
extern "C" {
#endif

// cudart-memory.cc
cudaError_t dce_cudaFree(void *devPtr);
cudaError_t dce_cudaFreeArray(cudaArray *array);
cudaError_t dce_cudaFreeHost(void *ptr);
cudaError_t dce_cudaGetSymbolAddress(void **devPtr, const void *symbol);
cudaError_t dce_cudaGetSymbolSize(size_t *size, const void *symbol);
cudaError_t dce_cudaHostAlloc(void **ptr, size_t size, unsigned int flags);
cudaError_t dce_cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags);
cudaError_t dce_cudaHostGetFlags(unsigned int *pFlags, void *pHost);
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
cudaError_t dce_cudaMalloc(void **devPtr, size_t size);
cudaError_t dce_cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
        cudaExtent extent);
#if CUDART_VERSION >= 3000
cudaError_t dce_cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent,
        unsigned int flags);
#else
cudaError_t dce_cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent);
#endif
#if CUDART_VERSION >= 3000
cudaError_t dce_cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags);
#else
cudaError_t dce_cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height);
#endif
cudaError_t dce_cudaMallocHost(void **ptr, size_t size);
cudaError_t dce_cudaMallocPitch(void **devPtr,
        size_t *pitch, size_t width, size_t height);
cudaError_t dce_cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpy2DArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t stream);
cudaError_t dce_cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t dce_cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t dce_cudaMemcpy3D(const cudaMemcpy3DParms *p);
cudaError_t dce_cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
        cudaStream_t stream);
cudaError_t dce_cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t count,
        cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t dce_cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream);
cudaError_t dce_cudaMemcpyFromSymbol(void *dst, const void *symbol,
        size_t count, size_t offset,
        cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t dce_cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream);
cudaError_t dce_cudaMemcpyToSymbol(const void *symbol, const void *src,
        size_t count, size_t offset,
        cudaMemcpyKind kind);
cudaError_t dce_cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t dce_cudaMemset(void *devPtr, int c, size_t count);
cudaError_t dce_cudaMemset2D(void *mem, size_t pitch, int c, size_t width,
        size_t height);
cudaError_t dce_cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
        cudaExtent extent);

// cudart-device.cc
cudaError_t dce_cudaChooseDevice(int *device, const cudaDeviceProp *prop);
cudaError_t dce_cudaGetDevice(int *device);
cudaError_t dce_cudaGetDeviceCount(int *count);
cudaError_t dce_cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
cudaError_t dce_cudaSetDevice(int device);
#if CUDART_VERSION >= 3000
cudaError_t dce_cudaSetDeviceFlags(unsigned int flags);
#else
cudaError_t dce_cudaSetDeviceFlags(int flags);
#endif
cudaError_t dce_cudaSetValidDevices(int *device_arr, int len);
cudaError_t dce_cudaDeviceReset (void);

// cudart-error.cc
const char* dce_cudaGetErrorString(cudaError_t error);
cudaError_t dce_cudaGetLastError(void);

// cudart-event.cc
cudaError_t dce_cudaEventCreate(cudaEvent_t *event);
#if CUDART_VERSION >= 3000
cudaError_t dce_cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags);
#else
cudaError_t dce_cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
#endif
cudaError_t dce_cudaEventDestroy(cudaEvent_t event);
cudaError_t dce_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t dce_cudaEventQuery(cudaEvent_t event);
cudaError_t dce_cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t dce_cudaEventSynchronize(cudaEvent_t event);

// cudart-execution.cc
cudaError_t dce_cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream);
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
cudaError_t dce_cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
        const void *func);
#endif
cudaError_t dce_cudaLaunch(const void *entry);
cudaError_t dce_cudaSetDoubleForDevice(double *d);
cudaError_t dce_cudaSetDoubleForHost(double *d);
cudaError_t dce_cudaSetupArgument(const void *arg, size_t size,
        size_t offset);

// cudart-internal.cc
void** dce___cudaRegisterFatBinary(void *fatCubin);
void dce___cudaUnregisterFatBinary(void **fatCubinHandle);
void dce___cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
void dce___cudaRegisterVar(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global);
void dce___cudaRegisterShared(void **fatCubinHandle, void **devicePtr);
void dce___cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
        size_t size, size_t alignment, int storage);
void dce___cudaRegisterTexture(void **fatCubinHandle,
        const textureReference *hostVar, void **deviceAddress, char *deviceName,
        int dim, int norm, int ext);
int dce___cudaSynchronizeThreads(void** x, void* y);
void dce___cudaTextureFetch(const void *tex, void *index, int integer,
        void *val);

// cudart-stream.cc
cudaError_t dce_cudaStreamCreate(cudaStream_t *pStream);
cudaError_t dce_cudaStreamDestroy(cudaStream_t stream);
cudaError_t dce_cudaStreamQuery(cudaStream_t stream);
cudaError_t dce_cudaStreamSynchronize(cudaStream_t stream);

// cudart-texture.cc
cudaError_t dce_cudaBindTexture(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t size);
cudaError_t dce_cudaBindTexture2D(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch);
cudaError_t dce_cudaBindTextureToArray(const textureReference *texref,
        const cudaArray *array, const cudaChannelFormatDesc *desc);
cudaChannelFormatDesc dce_cudaCreateChannelDesc(int x, int y, int z, int w,
        cudaChannelFormatKind f);
cudaError_t dce_cudaGetChannelDesc(cudaChannelFormatDesc *desc,
        const cudaArray *array);
cudaError_t dce_cudaGetTextureAlignmentOffset(size_t *offset,
        const textureReference *texref);
cudaError_t dce_cudaGetTextureReference(const textureReference **texref,
        const void *symbol);
cudaError_t dce_cudaUnbindTexture(const textureReference *texref);

// cudart-thread.cc
cudaError_t dce_cudaThreadSynchronize();
cudaError_t dce_cudaThreadExit();

// cudart-version.cc
cudaError_t dce_cudaDriverGetVersion(int *driverVersion);
cudaError_t dce_cudaRuntimeGetVersion(int *runtimeVersion);

#ifdef __cplusplus
}
#endif

#endif	/* _CUDART_H */

