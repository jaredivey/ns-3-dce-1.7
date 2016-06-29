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

/*Load a module's data. */
CUresult dce_cuModuleLoadData(CUmodule *module, const void *image) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddStringForArguments((char *) image, nodeId);
    CudaDrFrontend::Execute("cuModuleLoadData", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId))
        *module = (CUmodule) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));

}

/*Returns a function handle*/
CUresult dce_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddStringForArguments((char*) name, nodeId);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hmod, nodeId);
    CudaDrFrontend::Execute("cuModuleGetFunction", NULL, nodeId);
    void* tmp;
    if (CudaDrFrontend::Success(nodeId)) {
        tmp = (CUfunction) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
        *hfunc = (CUfunction) tmp;
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns a global pointer from a module.*/
CUresult dce_cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddStringForArguments((char*) name, nodeId);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hmod, nodeId);
    CudaDrFrontend::Execute("cuModuleGetGlobal", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId)) {
        *dptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
        *bytes = (size_t) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Returns a handle to a texture-reference.*/
CUresult dce_cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    CudaDrFrontend::AddStringForArguments((char*) name, nodeId);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hmod, nodeId);
    CudaDrFrontend::Execute("cuModuleGetTexRef", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId)) {
        *pTexRef = (CUtexref) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Load a module's data with options.*/
CUresult dce_cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaDrFrontend::Prepare(nodeId);
    char *tmp;
    char *tmp2;
    CudaDrFrontend::AddVariableForArguments(numOptions, nodeId);
    CudaDrFrontend::AddHostPointerForArguments(options, numOptions, nodeId);
    CudaDrFrontend::AddStringForArguments((char *) image, nodeId);
    for (unsigned int i = 0; i < numOptions; i++) {
        CudaDrFrontend::AddHostPointerForArguments(&optionValues[i], 1, nodeId);
    }
    CudaDrFrontend::Execute("cuModuleLoadDataEx", NULL, nodeId);
    if (CudaDrFrontend::Success(nodeId)) {
        *module = (CUmodule) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
        int len_str;
        for (unsigned int i = 0; i < numOptions; i++) {

            switch (options[i]) {
                case CU_JIT_INFO_LOG_BUFFER:
                    len_str = CudaDrFrontend::GetOutputVariable<int>(nodeId);
                    tmp = CudaDrFrontend::GetOutputHostPointer<char>(len_str, nodeId);
                    if (tmp) strcpy((char *) *(optionValues + i), tmp);
                    break;
                case CU_JIT_ERROR_LOG_BUFFER:
                    tmp2 = (CudaDrFrontend::GetOutputString(nodeId));
                    if (tmp) strcpy((char *) *(optionValues + i), tmp2);
                    break;

                default:
                    *(optionValues + i) = (void *) (*(CudaDrFrontend::GetOutputHostPointer<unsigned int>(1, nodeId)));
            }
        }
    }
    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

CUresult dce_cuModuleLoad(CUmodule *module, const char *fname) {
    // FIXME: implement
    cerr << "*** Error: cuModuleLoad not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    // FIXME: implement
    cerr << "*** Error: cuModuleLoadFatBinary not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuModuleUnload(CUmodule hmod) {
    // FIXME: implement
    cerr << "*** Error: cuModuleUnload not yet implemented!" << endl;
    return (CUresult) 1;
}
