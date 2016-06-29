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

#include "cudart.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

using namespace std;
using namespace ns3;

cudaError_t dce_cudaFree(void *devPtr) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddDevicePointerForArguments(devPtr, nodeId);
    CudaRtFrontend::Execute("cudaFree", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaFreeArray(cudaArray *array) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddDevicePointerForArguments((void *) array, nodeId);
    CudaRtFrontend::Execute("cudaFreeArray", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaFreeHost(void *ptr) {
	uint32_t nodeId = UtilsGetNodeId ();
    free(ptr);
    return cudaSuccess;
}

cudaError_t dce_cudaGetSymbolAddress(void **devPtr, const void *symbol) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    // Achtung: skip adding devPtr
    CudaRtFrontend::AddSymbolForArguments(symbol, nodeId);
    CudaRtFrontend::Execute("cudaGetSymbolAddress", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        *devPtr = CudaUtil::UnmarshalPointer(CudaRtFrontend::GetOutputString(nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaGetSymbolSize(size_t *size, const void *symbol) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(size, 1, nodeId);
    CudaRtFrontend::AddSymbolForArguments(symbol, nodeId);
    CudaRtFrontend::Execute("cudaGetSymbolSize", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        *size = *(CudaRtFrontend::GetOutputHostPointer<size_t > (1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
	uint32_t nodeId = UtilsGetNodeId ();
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

cudaError_t dce_cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
	uint32_t nodeId = UtilsGetNodeId ();
    // Achtung: we can't use mapped memory
    return cudaErrorMemoryAllocation;
}

cudaError_t dce_cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
	uint32_t nodeId = UtilsGetNodeId ();
    // Achtung: falling back to the simplest method because we can't map memory
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
    *pFlags = cudaHostAllocDefault;
#endif
    return cudaSuccess;
}

cudaError_t dce_cudaMalloc(void **devPtr, size_t size) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddVariableForArguments(size, nodeId);
    CudaRtFrontend::Execute("cudaMalloc", NULL, nodeId);

    if (CudaRtFrontend::Success(nodeId))
        *devPtr = CudaRtFrontend::GetOutputDevicePointer(nodeId);

    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
        cudaExtent extent) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

#if CUDART_VERSION >= 3000
cudaError_t dce_cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent,
        unsigned int flags) {
#else
cudaError_t dce_cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent) {
#endif
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3DArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

// FIXME: new mapping way

#if CUDART_VERSION >= 3000
cudaError_t dce_cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
#else
cudaError_t dce_cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height) {
#endif
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);

    CudaRtFrontend::AddHostPointerForArguments(desc, 1, nodeId);
    CudaRtFrontend::AddVariableForArguments(width, nodeId);
    CudaRtFrontend::AddVariableForArguments(height, nodeId);
    CudaRtFrontend::Execute("cudaMallocArray", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        *arrayPtr = (cudaArray *) CudaRtFrontend::GetOutputDevicePointer(nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMallocHost(void **ptr, size_t size) {
	uint32_t nodeId = UtilsGetNodeId ();
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

cudaError_t dce_cudaMallocPitch(void **devPtr,
        size_t *pitch, size_t width, size_t height) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);

    CudaRtFrontend::AddVariableForArguments(*pitch, nodeId);
    CudaRtFrontend::AddVariableForArguments(width, nodeId);
    CudaRtFrontend::AddVariableForArguments(height, nodeId);
    CudaRtFrontend::Execute("cudaMallocPitch", NULL, nodeId);

    if (CudaRtFrontend::Success(nodeId)) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer(nodeId);
        *pitch = CudaRtFrontend::GetOutputVariable<size_t>(nodeId);
    }
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst, nodeId);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpy", NULL, nodeId);
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("", 1, nodeId);
            CudaRtFrontend::AddDevicePointerForArguments(src, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpy", NULL, nodeId);
            if (CudaRtFrontend::Success(nodeId))
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count, nodeId), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst, nodeId);
            CudaRtFrontend::AddDevicePointerForArguments(src, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpy", NULL, nodeId);
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy2DArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DArrayToArray() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy3D(const cudaMemcpy3DParms *p) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
        cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t count,
        cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyArrayToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            CudaRtFrontend::AddHostPointerForArguments("", 1, nodeId);
            CudaRtFrontend::AddHostPointerForArguments("", 1, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
            CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync", NULL, nodeId);
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst, nodeId);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
            CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync", NULL, nodeId);
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("", 1, nodeId);
            CudaRtFrontend::AddDevicePointerForArguments(src, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
            CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync", NULL, nodeId);
            if (CudaRtFrontend::Success(nodeId))
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count, nodeId), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst, nodeId);
            CudaRtFrontend::AddDevicePointerForArguments(src, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
            CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync", NULL, nodeId);
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArrayAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpyFromSymbol(void *dst, const void *symbol,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToHost:
            // Achtung: adding a fake host pointer 
            CudaRtFrontend::AddDevicePointerForArguments((void *) 0x666, nodeId);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol), nodeId);
            CudaRtFrontend::AddStringForArguments((char*)symbol, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(offset, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol", NULL, nodeId);
            if (CudaRtFrontend::Success(nodeId))
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count, nodeId), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst, nodeId);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol), nodeId);
            CudaRtFrontend::AddStringForArguments((char*)symbol, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(offset, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol", NULL, nodeId);
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromSymbolAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *) dst, nodeId);
            CudaRtFrontend::AddVariableForArguments(wOffset, nodeId);
            CudaRtFrontend::AddVariableForArguments(hOffset, nodeId);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpyToArray", NULL, nodeId);
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *) dst, nodeId);
            CudaRtFrontend::AddVariableForArguments(wOffset, nodeId);
            CudaRtFrontend::AddVariableForArguments(hOffset, nodeId);
            CudaRtFrontend::AddDevicePointerForArguments(src, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpyToArray", NULL, nodeId);
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemcpyToSymbol(const void *symbol, const void *src,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol), nodeId);
            CudaRtFrontend::AddStringForArguments((char*)symbol, nodeId);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(offset, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol", NULL, nodeId);
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol), nodeId);
            CudaRtFrontend::AddStringForArguments((char*)symbol, nodeId);
            CudaRtFrontend::AddDevicePointerForArguments(src, nodeId);
            CudaRtFrontend::AddVariableForArguments(count, nodeId);
            CudaRtFrontend::AddVariableForArguments(kind, nodeId);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol", NULL, nodeId);
            break;
        case cudaMemcpyDefault:
        default:
            break;
    }

    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemset(void *devPtr, int c, size_t count) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddDevicePointerForArguments(devPtr, nodeId);
    CudaRtFrontend::AddVariableForArguments(c, nodeId);
    CudaRtFrontend::AddVariableForArguments(count, nodeId);
    CudaRtFrontend::Execute("cudaMemset", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaMemset2D(void *mem, size_t pitch, int c, size_t width,
        size_t height) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemset2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

cudaError_t dce_cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
        cudaExtent extent) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    cerr << "*** Error: cudaMemset3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}
