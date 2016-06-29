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

cudaError_t dce_cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(device, 1, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(prop, 1, nodeId);
    CudaRtFrontend::Execute("cudaChooseDevice", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *device = *(CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaGetDevice(int *device) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(device, 1, nodeId);
    CudaRtFrontend::Execute("cudaGetDevice", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *device = *(CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaGetDeviceCount(int *count) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(count, 1, nodeId);
    CudaRtFrontend::Execute("cudaGetDeviceCount", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *count = *(CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(prop, 1, nodeId);
    CudaRtFrontend::AddVariableForArguments(device, nodeId);
    CudaRtFrontend::Execute("cudaGetDeviceProperties", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId)) {
        memmove(prop, CudaRtFrontend::GetOutputHostPointer<cudaDeviceProp>(1, nodeId),
                sizeof(cudaDeviceProp));
        std::cout << "GPU Device " << device <<
        		": \"" << prop->name << "\" with compute capability " << prop->major << "." << prop->minor << std::endl;
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
        prop->canMapHostMemory = 0;
#endif
    }
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaSetDevice(int device) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddVariableForArguments(device, nodeId);
    CudaRtFrontend::Execute("cudaSetDevice", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

#if CUDART_VERSION >= 3000
cudaError_t dce_cudaSetDeviceFlags(unsigned int flags) {
#else
cudaError_t dce_cudaSetDeviceFlags(int flags) {
#endif
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddVariableForArguments(flags, nodeId);
    CudaRtFrontend::Execute("cudaSetDeviceFlags", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaSetValidDevices(int *device_arr, int len) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(device_arr, len, nodeId);
    CudaRtFrontend::AddVariableForArguments(len, nodeId);
    CudaRtFrontend::Execute("cudaSetValidDevices", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId)) {
        int *out_device_arr = CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId);
        memmove(device_arr, out_device_arr, sizeof(int) * len);
    }
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaDeviceReset(void) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::Execute("cudaDeviceReset", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}
