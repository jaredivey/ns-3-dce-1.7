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

/*Creates an event.*/
CUresult dce_cuEventCreate(CUevent *phEvent, unsigned int Flags) {
	return cuEventCreate(phEvent, Flags);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(Flags, nodeId);
//    CudaDrFrontend::Execute("cuEventCreate", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *phEvent = (CUevent) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Destroys an event.*/
CUresult dce_cuEventDestroy(CUevent hEvent) {
	return cuEventDestroy(hEvent);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hEvent, nodeId);
//    CudaDrFrontend::Execute("cuEventDestroy", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Computes the elapsed time between two events.*/
CUresult dce_cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
	return cuEventElapsedTime(pMilliseconds, hStart, hEnd);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddHostPointerForArguments(pMilliseconds, 1, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hStart, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hEnd, nodeId);
//    CudaDrFrontend::Execute("cuEventElapsedTime", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pMilliseconds = *(CudaDrFrontend::GetOutputHostPointer<float>(1, nodeId));
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Queries an event's status.*/
CUresult dce_cuEventQuery(CUevent hEvent) {
	return cuEventQuery(hEvent);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hEvent, nodeId);
//    CudaDrFrontend::Execute("cuEventQuery", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Records an event.*/
CUresult dce_cuEventRecord(CUevent hEvent, CUstream hStream) {
	return cuEventRecord(hEvent,hStream);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hEvent, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hStream, nodeId);
//    CudaDrFrontend::Execute("cuEventRecord", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Waits for an event to complete.*/
CUresult dce_cuEventSynchronize(CUevent hEvent) {
	return cuEventSynchronize(hEvent);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hEvent, nodeId);
//    CudaDrFrontend::Execute("cuEventSynchronize", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}
