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

using namespace std;
using namespace ns3;

cudaError_t dce_cudaBindTexture(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t size) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(offset, 1, nodeId);
    // Achtung: passing the address and the content of the textureReference
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref), nodeId);
    CudaRtFrontend::AddHostPointerForArguments(texref, 1, nodeId);
    CudaRtFrontend::AddDevicePointerForArguments(devPtr, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(desc, 1, nodeId);
    CudaRtFrontend::AddVariableForArguments(size, nodeId);
    CudaRtFrontend::Execute("cudaBindTexture", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        *offset = *(CudaRtFrontend::GetOutputHostPointer<size_t > (1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaBindTexture2D(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(offset, 1, nodeId);
    // Achtung: passing the address and the content of the textureReference
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref), nodeId);
    CudaRtFrontend::AddHostPointerForArguments(texref, 1, nodeId);
    CudaRtFrontend::AddDevicePointerForArguments(devPtr, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(desc, 1, nodeId);
    CudaRtFrontend::AddVariableForArguments(width, nodeId);
    CudaRtFrontend::AddVariableForArguments(height, nodeId);
    CudaRtFrontend::AddVariableForArguments(pitch, nodeId);
    CudaRtFrontend::Execute("cudaBindTexture2D", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        *offset = *(CudaRtFrontend::GetOutputHostPointer<size_t > (1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaBindTextureToArray(const textureReference *texref,
        const cudaArray *array, const cudaChannelFormatDesc *desc) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    // Achtung: passing the address and the content of the textureReference
    CudaRtFrontend::AddVariableForArguments((uint64_t) texref, nodeId);
    CudaRtFrontend::AddDevicePointerForArguments((void *) array, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(desc, 1, nodeId);
    CudaRtFrontend::Execute("cudaBindTextureToArray", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaChannelFormatDesc dce_cudaCreateChannelDesc(int x, int y, int z, int w,
        cudaChannelFormatKind f) {
	uint32_t nodeId = UtilsGetNodeId ();
    cudaChannelFormatDesc desc;
    desc.x = x;
    desc.y = y;
    desc.z = z;
    desc.w = w;
    desc.f = f;
    return desc;
}

cudaError_t dce_cudaGetChannelDesc(cudaChannelFormatDesc *desc,
        const cudaArray *array) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(desc, 1, nodeId);
    CudaRtFrontend::AddDevicePointerForArguments(array, nodeId);
    CudaRtFrontend::Execute("cudaGetChannelDesc", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        memmove(desc, CudaRtFrontend::GetOutputHostPointer<cudaChannelFormatDesc > (1, nodeId),
            sizeof (cudaChannelFormatDesc));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaGetTextureAlignmentOffset(size_t *offset,
        const textureReference *texref) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(offset, 1, nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref), nodeId);
    CudaRtFrontend::Execute("cudaGetTextureAlignmentOffset", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        *offset = *(CudaRtFrontend::GetOutputHostPointer<size_t > (1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaGetTextureReference(const textureReference **texref,
        const void *symbol) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    // Achtung: skipping to add texref
    // Achtung: passing the address and the content of symbol
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol), nodeId);
    CudaRtFrontend::AddStringForArguments((char*)symbol, nodeId);
    CudaRtFrontend::Execute("cudaGetTextureReference", NULL, nodeId);
    if (CudaRtFrontend::Success(nodeId)) {
        char *texrefHandler = CudaRtFrontend::GetOutputString(nodeId);
        *texref = (textureReference *) strtoul(texrefHandler, NULL, 16);
    }
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaUnbindTexture(const textureReference *texref) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref), nodeId);
    CudaRtFrontend::Execute("cudaUnbindTexture", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}
