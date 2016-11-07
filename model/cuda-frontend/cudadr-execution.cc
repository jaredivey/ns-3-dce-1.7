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

/*Sets the parameter size for the function.*/
CUresult dce_cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
	return cuParamSetSize(hfunc, numbytes);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(numbytes, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuParamSetSize", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Sets the block-dimensions for the function.*/
CUresult dce_cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
	return cuFuncSetBlockShape(hfunc, x, y, z);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(x, nodeId);
//    CudaDrFrontend::AddVariableForArguments(y, nodeId);
//    CudaDrFrontend::AddVariableForArguments(z, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuFuncSetBlockShape", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Launches a CUDA function.*/
CUresult dce_cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
	return cuLaunchGrid(f, grid_width, grid_height);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(grid_width, nodeId);
//    CudaDrFrontend::AddVariableForArguments(grid_height, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) f, nodeId);
//    CudaDrFrontend::Execute("cuLaunchGrid", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Returns information about a function.*/
CUresult dce_cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
	return cuFuncGetAttribute(pi, attrib, hfunc);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddHostPointerForArguments(pi, 1, nodeId);
//    CudaDrFrontend::AddVariableForArguments(attrib, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuFuncGetAttribute", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pi = *(CudaDrFrontend::GetOutputHostPointer<int>(1, nodeId));
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Sets the dynamic shared-memory size for the function.*/
CUresult dce_cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
	return cuFuncSetSharedSize(hfunc, bytes);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(bytes, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuFuncSetSharedSize", NULL, nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Launches a CUDA function.*/
CUresult dce_cuLaunch(CUfunction f) {
	return cuLaunch(f);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) f, nodeId);
//    CudaDrFrontend::Execute("cuLaunch", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

CUresult dce_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
		unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
		unsigned int sharedMemBytes, CUstream hStream, void ** kernelParams, void ** extra)
{
	return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
//	uint32_t nodeId = UtilsGetNodeId ();
//	dce_cuFuncSetBlockShape(f, blockDimX, blockDimY, blockDimZ);
//	dce_cuFuncSetSharedSize(f, sharedMemBytes);
//	void **tmp = extra;
//	void *args = NULL;
//	unsigned long argsSize = 0;
//	while (*tmp != CU_LAUNCH_PARAM_END)
//	{
//		if (*tmp == CU_LAUNCH_PARAM_BUFFER_POINTER)
//		{
//			++tmp;
//			args = *(tmp);
//		}
//		if (*tmp == CU_LAUNCH_PARAM_BUFFER_SIZE)
//		{
//			++tmp;
//			memcpy (&argsSize, *tmp, sizeof(int));
//		}
//		++tmp;
//	}
//	if (args != NULL && argsSize != 0)
//	{
//		dce_cuParamSetv(f, 0, args, argsSize);
//	}
//	else
//	{
//		return (CUresult) 1;
//	}
//	return dce_cuLaunchGridAsync (f, gridDimX, gridDimY, hStream);
}

/*Launches a CUDA function.*/
CUresult dce_cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
	return cuLaunchGridAsync(f, grid_width, grid_height, hStream);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(grid_width, nodeId);
//    CudaDrFrontend::AddVariableForArguments(grid_height, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) f, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hStream, nodeId);
//    CudaDrFrontend::Execute("cuLaunchGridAsync", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Adds a floating-point parameter to the function's argument list.*/
CUresult dce_cuParamSetf(CUfunction hfunc, int offset, float value) {
	return cuParamSetf(hfunc, offset, value);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(offset, nodeId);
//    CudaDrFrontend::AddVariableForArguments(value, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuParamSetf", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Adds an integer parameter to the function's argument list.*/
CUresult dce_cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
	return cuParamSeti(hfunc, offset, value);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(offset, nodeId);
//    CudaDrFrontend::AddVariableForArguments(value, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuParamSeti", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Adds a texture-reference to the function's argument list.*/
CUresult dce_cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
	return cuParamSetTexRef(hfunc, texunit, hTexRef);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::AddVariableForArguments(texunit, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuParamSetTexRef", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Adds arbitrary data to the function's argument list.*/
CUresult dce_cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
	return cuParamSetv(hfunc,offset,ptr,numbytes);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(offset, nodeId);
//    CudaDrFrontend::AddVariableForArguments(numbytes, nodeId);
//    CudaDrFrontend::AddHostPointerForArguments((char *) ptr, numbytes, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuParamSetv", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Sets the preferred cache configuration for a device function. */
CUresult dce_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
	return cuFuncSetCacheConfig(hfunc,config);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(config, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hfunc, nodeId);
//    CudaDrFrontend::Execute("cuFuncSetCacheConfig", NULL, nodeId);
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}


