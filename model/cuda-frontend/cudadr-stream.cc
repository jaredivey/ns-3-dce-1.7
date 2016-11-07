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

/*Create a stream.*/
CUresult dce_cuStreamCreate(CUstream *phStream, unsigned int Flags) {
	return cuStreamCreate(phStream, Flags);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(Flags, nodeId);
//    CudaDrFrontend::Execute("cuStreamCreate", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *phStream = (CUstream) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Destroys a stream.*/
CUresult dce_cuStreamDestroy(CUstream hStream) {
	return cuStreamDestroy(hStream);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hStream, nodeId);
//    CudaDrFrontend::Execute("cuStreamDestroy", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Determine status of a compute stream.*/
CUresult dce_cuStreamQuery(CUstream hStream) {
	return cuStreamQuery(hStream);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hStream, nodeId);
//    CudaDrFrontend::Execute("cuStreamQuery", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Wait until a stream's tasks are completed.*/
CUresult dce_cuStreamSynchronize(CUstream hStream) {
	return cuStreamSynchronize(hStream);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hStream, nodeId);
//    CudaDrFrontend::Execute("cuStreamSynchronize", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}
