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

/*Create a CUDA context*/
CUresult dce_cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
	return cuCtxCreate(pctx,flags,dev);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(flags, nodeId);
//    CudaDrFrontend::AddVariableForArguments(dev, nodeId);
//    CudaDrFrontend::Execute("cuCtxCreate", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pctx = (CUcontext) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Increment a context's usage-count*/
CUresult dce_cuCtxAttach(CUcontext *pctx, unsigned int flags) {
	return cuCtxAttach(pctx,flags);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(flags, nodeId);
//    CudaDrFrontend::AddHostPointerForArguments(pctx, 1, nodeId);
//    CudaDrFrontend::Execute("cuCtxAttach", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pctx = (CUcontext) CudaDrFrontend::GetOutputDevicePointer(nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Destroy the current context or a floating CUDA context*/
CUresult dce_cuCtxDestroy(CUcontext ctx) {
	return cuCtxDestroy(ctx);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx, nodeId);
//    CudaDrFrontend::Execute("cuCtxDestroy", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Decrement a context's usage-count. */
CUresult dce_cuCtxDetach(CUcontext ctx) {
	return cuCtxDetach(ctx);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx, nodeId);
//    CudaDrFrontend::Execute("cuCtxDetach", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Returns the device ID for the current context.*/
CUresult dce_cuCtxGetDevice(CUdevice *device) {
	return cuCtxGetDevice(device);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddHostPointerForArguments(device, 1, nodeId);
//    CudaDrFrontend::Execute("cuCtxGetDevice", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId)) {
//        *device = *(CudaDrFrontend::GetOutputHostPointer<CUdevice > (1, nodeId));
//    }
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Pops the current CUDA context from the current CPU thread.*/
CUresult dce_cuCtxPopCurrent(CUcontext *pctx) {
	return cuCtxPopCurrent(pctx);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CUcontext ctx;
//    pctx = &ctx;
//    CudaDrFrontend::Execute("cuCtxPopCurrent", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pctx = (CUcontext) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Pushes a floating context on the current CPU thread. */
CUresult dce_cuCtxPushCurrent(CUcontext ctx) {
	return cuCtxPushCurrent(ctx);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx, nodeId);
//    CudaDrFrontend::Execute("cuCtxPushCurrent", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Block for a context's tasks to complete.*/
CUresult dce_cuCtxSynchronize(void) {
	return cuCtxSynchronize();
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::Execute("cuCtxSynchronize", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}
