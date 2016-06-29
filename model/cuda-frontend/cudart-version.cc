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

cudaError_t dce_cudaDriverGetVersion(int *driverVersion) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(driverVersion, 1, nodeId);
    CudaRtFrontend::Execute("cudaDriverGetVersion", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *driverVersion = *(CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}

cudaError_t dce_cudaRuntimeGetVersion(int *runtimeVersion) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddHostPointerForArguments(runtimeVersion, 1, nodeId);
    CudaRtFrontend::Execute("cudaDriverGetVersion", NULL, nodeId);
    if(CudaRtFrontend::Success(nodeId))
        *runtimeVersion = *(CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId));
    return CudaRtFrontend::GetExitCode(nodeId);
}
