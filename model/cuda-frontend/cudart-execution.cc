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

cudaError_t dce_cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream) {
	uint32_t nodeId = ns3::UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
#if 0
    CudaRtFrontend::AddVariableForArguments(gridDim, nodeId);
    CudaRtFrontend::AddVariableForArguments(blockDim, nodeId);
    CudaRtFrontend::AddVariableForArguments(sharedMem, nodeId);
    CudaRtFrontend::AddVariableForArguments(stream, nodeId);
    CudaRtFrontend::Execute("cudaConfigureCall", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
#endif
    Buffer *launch = CudaRtFrontend::GetLaunchBuffer(nodeId);
    launch->Reset();
    // CNCL
    launch->Add<int>(0x434e34c);
    launch->Add(gridDim);
    launch->Add(blockDim);
    launch->Add(sharedMem);
#if CUDART_VERSION >= 3010
    launch->Add((uint64_t) stream);
#else
    launch->Add(stream);
#endif
    return cudaSuccess;
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
cudaError_t dce_cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
        const void *func) {
	uint32_t nodeId = ns3::UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(attr, 1, nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(func), nodeId);
    CudaRtFrontend::Execute("cudaFuncGetAttributes", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        memmove(attr, CudaRtFrontend::GetOutputHostPointer<cudaFuncAttributes>(1, nodeId),
                sizeof(cudaFuncAttributes));
    return CudaRtFrontend::GetExitCode(nodeId);
}
#endif

cudaError_t dce_cudaLaunch(const void *entry) {
	uint32_t nodeId = ns3::UtilsGetNodeId ();
#if 0
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(entry), nodeId);
    CudaRtFrontend::Execute("cudaLaunch", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
#endif
    Buffer *launch = CudaRtFrontend::GetLaunchBuffer(nodeId);
    // LAUN
    launch->Add<int>(0x4c41554e);
    launch->AddString(CudaUtil::MarshalHostPointer(entry));
    CudaRtFrontend::Execute("cudaLaunch", launch, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaSetDoubleForDevice(double *d) {
	uint32_t nodeId = ns3::UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(d, 1, nodeId);
    CudaRtFrontend::Execute("cudaSetDoubleForDevice", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *d = *(CudaRtFrontend::GetOutputHostPointer<double >(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaSetDoubleForHost(double *d) {
	uint32_t nodeId = ns3::UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(d, 1, nodeId);
    CudaRtFrontend::Execute("cudaSetDoubleForHost", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *d = *(CudaRtFrontend::GetOutputHostPointer<double >(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaSetupArgument(const void *arg, size_t size,
        size_t offset) {
	uint32_t nodeId = ns3::UtilsGetNodeId ();
#if 0
    CudaRtFrontend::AddHostPointerForArguments(static_cast<const char *> (arg), size, nodeId);
    CudaRtFrontend::AddVariableForArguments(size, nodeId);
    CudaRtFrontend::AddVariableForArguments(offset, nodeId);
    CudaRtFrontend::Execute("cudaSetupArgument", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
#endif
    Buffer *launch = CudaRtFrontend::GetLaunchBuffer(nodeId);
    // STAG
    launch->Add<int>(0x53544147);
    launch->Add<char>(static_cast<char *>(const_cast<void *>(arg)), size);
    launch->Add(size);
    launch->Add(offset);
    return cudaSuccess;
}
