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

const char* dce_cudaGetErrorString(cudaError_t error) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddVariableForArguments(error, nodeId);
    CudaRtFrontend::Execute("cudaGetErrorString", NULL, nodeId);
#ifdef _WIN32
    char *error_string = _strdup(CudaRtFrontend::GetOutputString(nodeId));
#else
    char *error_string = strdup(CudaRtFrontend::GetOutputString(nodeId));
#endif
    return error_string;
}

cudaError_t dce_cudaGetLastError(void) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::Execute("cudaGetLastError", NULL, nodeId);
    return CudaRtFrontend::GetExitCode(nodeId);
}
