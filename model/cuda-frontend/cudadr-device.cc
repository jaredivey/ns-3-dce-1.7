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

/*Returns the compute capability of the device*/
CUresult dce_cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddHostPointerForArguments(major, 1, nodeId);
    CudaDrFrontend::AddHostPointerForArguments(minor, 1, nodeId);
    CudaDrFrontend::AddVariableForArguments(dev, nodeId);
    CudaDrFrontend::Execute("cuDeviceComputeCapability", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId)) {
        *major = *(CudaDrFrontend::GetOutputHostPointer<int>(1, nodeId));
        *minor = *(CudaDrFrontend::GetOutputHostPointer<int>(1, nodeId));
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns a handle to a compute device*/
CUresult dce_cuDeviceGet(CUdevice *device, int ordinal) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddHostPointerForArguments(device, 1, nodeId);
    CudaDrFrontend::AddVariableForArguments(ordinal, nodeId);
    CudaDrFrontend::Execute("cuDeviceGet", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId)) {
        *device = *(CudaDrFrontend::GetOutputHostPointer<CUdevice > (1, nodeId));
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns information about the device*/
CUresult dce_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddHostPointerForArguments(pi, 1, nodeId);
    CudaDrFrontend::AddVariableForArguments(attrib, nodeId);
    CudaDrFrontend::AddVariableForArguments(dev, nodeId);
    CudaDrFrontend::Execute("cuDeviceGetAttribute", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId))
        *pi = *(CudaDrFrontend::GetOutputHostPointer<int>(1, nodeId));
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns the number of compute-capable devices. */
CUresult dce_cuDeviceGetCount(int *count) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddHostPointerForArguments(count, 1, nodeId);
    CudaDrFrontend::Execute("cuDeviceGetCount", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId))
        *count = *(CudaDrFrontend::GetOutputHostPointer<int>(1, nodeId));
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns an identifer string for the device.*/
CUresult dce_cuDeviceGetName(char *name, int len, CUdevice dev) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddStringForArguments((char *) name, nodeId);
    CudaDrFrontend::AddVariableForArguments(len, nodeId);
    CudaDrFrontend::AddVariableForArguments(dev, nodeId);
    char *temp = NULL;
    CudaDrFrontend::Execute("cuDeviceGetName", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId))
        temp = (CudaDrFrontend::GetOutputString(nodeId));
    strcpy(name, temp);
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns properties for a selected device.*/
CUresult dce_cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddHostPointerForArguments(prop, 1, nodeId);
    CudaDrFrontend::AddVariableForArguments(dev, nodeId);
    CudaDrFrontend::Execute("cuDeviceGetProperties", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId)) {
        memmove(prop, CudaDrFrontend::GetOutputHostPointer<CUdevprop > (1, nodeId), sizeof (CUdevprop));
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns the total amount of memory on the device. */
CUresult dce_cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddHostPointerForArguments(bytes, 1, nodeId);
    CudaDrFrontend::AddVariableForArguments(dev, nodeId);
    CudaDrFrontend::Execute("cuDeviceTotalMem", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId))
        *bytes = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}
