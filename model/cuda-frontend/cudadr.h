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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#ifndef CUDADR_H_
#define CUDADR_H_

#include <cstring>
#include <stdio.h>

#include </usr/local/cuda/include/host_defines.h>
#include </usr/local/cuda/include/builtin_types.h>
#include </usr/local/cuda/include/driver_types.h>
#include </usr/local/cuda/include/cuda.h>

#include "ns3/utils.h"
#include "ns3/cuda-util.h"
#include "cudadr-frontend.h"

#define __dv(v)

#ifdef __cplusplus
extern "C" {
#endif

// cudadr-context.cc
CUresult dce_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult dce_cuCtxAttach(CUcontext *pctx, unsigned int flags);
CUresult dce_cuCtxDestroy(CUcontext ctx);
CUresult dce_cuCtxDetach(CUcontext ctx);
CUresult dce_cuCtxGetDevice(CUdevice *device);
CUresult dce_cuCtxPopCurrent(CUcontext *pctx);
CUresult dce_cuCtxPushCurrent(CUcontext ctx);
CUresult dce_cuCtxSynchronize(void);

// cudadr-device.cc
CUresult dce_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
CUresult dce_cuDeviceGet(CUdevice *device, int ordinal);
CUresult dce_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult dce_cuDeviceGetCount(int *count);
CUresult dce_cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult dce_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev);
CUresult dce_cuDeviceTotalMem(size_t *bytes, CUdevice dev);

// cudadr-event.cc
CUresult dce_cuEventCreate(CUevent *phEvent, unsigned int Flags);
CUresult dce_cuEventDestroy(CUevent hEvent);
CUresult dce_cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
CUresult dce_cuEventQuery(CUevent hEvent);
CUresult dce_cuEventRecord(CUevent hEvent, CUstream hStream);
CUresult dce_cuEventSynchronize(CUevent hEvent);

// cudadr-execution.cc
CUresult dce_cuParamSetSize(CUfunction hfunc, unsigned int numbytes);
CUresult dce_cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z);
CUresult dce_cuLaunchGrid(CUfunction f, int grid_width, int grid_height);
CUresult dce_cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
CUresult dce_cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes);
CUresult dce_cuLaunch(CUfunction f);
CUresult dce_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
		unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
		unsigned int sharedMemBytes, CUstream hStream, void ** kernelParams, void ** extra);
CUresult dce_cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream);
CUresult dce_cuParamSetf(CUfunction hfunc, int offset, float value);
CUresult dce_cuParamSeti(CUfunction hfunc, int offset, unsigned int value);
CUresult dce_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef);
CUresult dce_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes);
CUresult dce_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);

// cudadr-initialization.cc
CUresult dce_cuInit(unsigned int flags);

// cudadr-memory.cc
CUresult dce_cuMemFree(CUdeviceptr dptr);
CUresult dce_cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult dce_cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult dce_cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult dce_cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
CUresult dce_cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
CUresult dce_cuMemcpy2D(const CUDA_MEMCPY2D *pCopy);
CUresult dce_cuArrayDestroy(CUarray hArray);
CUresult dce_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
CUresult dce_cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
CUresult dce_cuMemGetInfo(size_t *free, size_t *total);

// cudadr-module.cc
CUresult dce_cuModuleLoadData(CUmodule *module, const void *image);
CUresult dce_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult dce_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
CUresult dce_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
CUresult dce_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);

// cudadr-stream.cc
CUresult dce_cuStreamCreate(CUstream *phStream, unsigned int Flags);
CUresult dce_cuStreamDestroy(CUstream hStream);
CUresult dce_cuStreamQuery(CUstream hStream);
CUresult dce_cuStreamSynchronize(CUstream hStream);

// cudadr-texture.cc
CUresult dce_cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
CUresult dce_cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
CUresult dce_cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
CUresult dce_cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
CUresult dce_cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
CUresult dce_cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef);
CUresult dce_cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef);
CUresult dce_cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef);
CUresult dce_cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);

// cudadr-version.cc
CUresult dce_cuDriverGetVersion(int *driverVersion);

#ifdef __cplusplus
}
#endif

#endif /* CUDADR_H_ */
