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

cudaError_t dce_cudaEventCreate(cudaEvent_t *event) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::Execute("cudaEventCreate", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *event = (cudaEvent_t) CudaRtFrontend::GetOutputDevicePointer(nodeId);
#else
    CudaRtFrontend::AddHostPointerForArguments((void*)event, 1, nodeId);
    CudaRtFrontend::Execute("cudaEventCreate", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *event = *(CudaRtFrontend::GetOutputHostPointer<cudaEvent_t>(1, nodeId));
#endif
    return CudaRtFrontend::GetExitCode(nodeId);
}

#if CUDART_VERSION >= 3000
cudaError_t dce_cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags) {
#else
cudaError_t dce_cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
#endif
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddVariableForArguments(flags, nodeId);
    CudaRtFrontend::Execute("cudaEventCreateWithFlags", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *event = (cudaEvent_t) CudaRtFrontend::GetOutputDevicePointer(nodeId);
#else
    CudaRtFrontend::AddHostPointerForArguments((void*)event, 1, nodeId);
    CudaRtFrontend::AddVariableForArguments(flags, nodeId);
    CudaRtFrontend::Execute("cudaEventCreateWithFlags", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *event = *(CudaRtFrontend::GetOutputHostPointer<cudaEvent_t>(1, nodeId));
#endif
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaEventDestroy(cudaEvent_t event) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void *)event, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(event, nodeId);
#endif
    CudaRtFrontend::Execute("cudaEventDestroy", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(ms, nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)start, nodeId);
    CudaRtFrontend::AddDevicePointerForArguments((void*)end, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(start, nodeId);
    CudaRtFrontend::AddVariableForArguments(end, nodeId);
#endif
    CudaRtFrontend::Execute("cudaEventElapsedTime", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *ms = *(CudaRtFrontend::GetOutputHostPointer<float>(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaEventQuery(cudaEvent_t event) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)event, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(event, nodeId);
#endif
    CudaRtFrontend::Execute("cudaEventQuery", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)event, nodeId);
    CudaRtFrontend::AddDevicePointerForArguments((void*)stream, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(event, nodeId);
    CudaRtFrontend::AddVariableForArguments(stream, nodeId);
#endif
    CudaRtFrontend::Execute("cudaEventRecord", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaEventSynchronize(cudaEvent_t event) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments((void*)event, nodeId);
#else
    CudaRtFrontend::AddVariableForArguments(event, nodeId);
#endif    
    CudaRtFrontend::Execute("cudaEventSynchronize", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}
