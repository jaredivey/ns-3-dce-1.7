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

cudaError_t dce_cudaStreamCreate(cudaStream_t *pStream) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::Execute("cudaStreamCreate", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *pStream = (cudaStream_t) CudaRtFrontend::GetOutputDevicePointer(nodeId);
#else
    CudaRtFrontend::AddHostPointerForArguments(pStream, 1, nodeId);
    CudaRtFrontend::Execute("cudaStreamCreate", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *pStream = *(CudaRtFrontend::GetOutputHostPointer<cudaStream_t>(1, nodeId));
#endif
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaStreamDestroy(cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif    
    CudaRtFrontend::Execute("cudaStreamDestroy", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaStreamQuery(cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif     
    CudaRtFrontend::Execute("cudaStreamQuery", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaStreamSynchronize(cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif     
    CudaRtFrontend::Execute("cudaStreamSynchronize", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}
