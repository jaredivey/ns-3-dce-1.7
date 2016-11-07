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

#include "cudadr.h"

using namespace std;
using namespace ns3;

/*Frees device memory.*/
CUresult dce_cuMemFree(CUdeviceptr dptr) {
	return cuMemFree(dptr);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(dptr, nodeId);
//    CudaDrFrontend::Execute("cuMemFree", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Allocates device memory.*/
CUresult dce_cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
	return cuMemAlloc(dptr, bytesize);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(bytesize, nodeId);
//    CudaDrFrontend::Execute("cuMemAlloc", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *dptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Copies memory from Device to Host. */
CUresult dce_cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
	return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(srcDevice, nodeId);
//    CudaDrFrontend::AddVariableForArguments(ByteCount, nodeId);
//    CudaDrFrontend::Execute("cuMemcpyDtoH", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        memmove(dstHost, CudaDrFrontend::GetOutputHostPointer<char>(ByteCount, nodeId), ByteCount);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Copies memory from Host to Device.*/
CUresult dce_cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
	return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(ByteCount, nodeId);
//    CudaDrFrontend::AddVariableForArguments(dstDevice, nodeId);
//    CudaDrFrontend::AddHostPointerForArguments<char>(static_cast<char *>
//            (const_cast<void *> (srcHost)), ByteCount, nodeId);
//    CudaDrFrontend::Execute("cuMemcpyHtoD", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);

}

/*Creates a 1D or 2D CUDA array. */
CUresult dce_cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
	return cuArrayCreate(pHandle, pAllocateArray);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddHostPointerForArguments(pAllocateArray, 1, nodeId);
//    CudaDrFrontend::Execute("cuArrayCreate", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pHandle = (CUarray) CudaDrFrontend::GetOutputDevicePointer(nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Creates a 3D CUDA array.*/
CUresult dce_cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
	return cuArray3DCreate(pHandle, pAllocateArray);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddHostPointerForArguments(pAllocateArray, 1, nodeId);
//    CudaDrFrontend::Execute("cuArrayCreate", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pHandle = (CUarray) CudaDrFrontend::GetOutputDevicePointer(nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Copies memory for 2D arrays.*/
CUresult dce_cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
	return cuMemcpy2D(pCopy);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    int flag = 0;
//    CudaDrFrontend::AddHostPointerForArguments(pCopy, 1, nodeId);
//    if (pCopy->srcHost != NULL) {
//        flag = 1;
//        CudaDrFrontend::AddVariableForArguments(flag, nodeId);
//        CudaDrFrontend::AddHostPointerForArguments((char*) pCopy->srcHost, (pCopy->WidthInBytes)*(pCopy->Height), nodeId);
//    } else {
//        flag = 0;
//        CudaDrFrontend::AddVariableForArguments(flag, nodeId);
//    }
//    CudaDrFrontend::Execute("cuMemcpy2D", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Destroys a CUDA array.*/
CUresult dce_cuArrayDestroy(CUarray hArray) {
	return cuArrayDestroy(hArray);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments(hArray, nodeId);
//    CudaDrFrontend::Execute("cuArrayDestroy", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Allocates pitched device memory.*/
CUresult dce_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
	return cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(WidthInBytes, nodeId);
//    CudaDrFrontend::AddVariableForArguments(Height, nodeId);
//    CudaDrFrontend::AddVariableForArguments(ElementSizeBytes, nodeId);
//    CudaDrFrontend::Execute("cuMemAllocPitch", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId)) {
//        *dptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//        *pPitch = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
//    }
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Get information on memory allocations.*/
CUresult dce_cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
	return cuMemGetAddressRange(pbase, psize, dptr);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(dptr, nodeId);
//    CudaDrFrontend::Execute("cuMemGetAddressRange", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId)) {
//        *pbase = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//        *psize = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
//    }
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Gets free and total memory.*/
CUresult dce_cuMemGetInfo(size_t *free, size_t *total) {
	return cuMemGetInfo(free, total);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::Execute("cuMemGetInfo", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId)) {
//        *free = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
//        *total = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
//    }
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

CUresult dce_cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    // FIXME: implement
    cerr << "*** Error: cuArray3DGetDescriptor() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    // FIXME: implement
    cerr << "*** Error: cuArrayGetDescriptor() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemAllocHost(void **pp, size_t bytesize) {
    // FIXME: implement
    cerr << "*** Error: cuMemAllocHost() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy2Dasync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy2DUnaligned() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy3D() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy3DAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyAtoA(CUarray dstArray, size_t dstIndex, CUarray srcArray, size_t srcIndex, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoA() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray hSrc, size_t SrcIndex, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoD() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcIndex, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoH() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t srcIndex, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoHAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyDtoA(CUarray dstArray, size_t dstIndex, CUdeviceptr srcDevice, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoA() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoD() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoDAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoHAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyHtoA(CUarray dstArray, size_t dstIndex, const void *pSrc, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyHtoA() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyHtoAAsync(CUarray dstArray, size_t dstIndex, const void *pSrc, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyHtoAAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyHtoDAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemFreeHost(void *p) {
    // FIXME: implement
    cerr << "*** Error: cuMemFreeHost() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    // FIXME: implement
    cerr << "*** Error: cuMemHostAlloc() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    // FIXME: implement
    cerr << "*** Error: cuMemHostGetDevicePointer() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    // FIXME: implement
    cerr << "*** Error: cuMemHostGetFlags() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD16() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD2D16() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD2D32() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD2D8() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD32() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD8() not yet implemented!" << endl;
    return (CUresult) 1;
}
