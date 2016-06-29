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

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

using namespace ns3;

void** dce___cudaRegisterFatBinary(void *fatCubin) {
	uint32_t nodeId = UtilsGetNodeId ();
    /* Fake host pointer */
    Buffer * input_buffer = new Buffer();
    input_buffer->AddString(CudaUtil::MarshalHostPointer((void **) fatCubin));
    input_buffer = CudaUtil::MarshalFatCudaBinary(
            (__cudaFatCudaBinary *) fatCubin, input_buffer);

    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::Execute("cudaRegisterFatBinary", input_buffer, nodeId);
    if (CudaRtFrontend::Success(nodeId))
        return (void **) fatCubin;
    return NULL;
}

void dce___cudaUnregisterFatBinary(void **fatCubinHandle) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle), nodeId);
    CudaRtFrontend::Execute("cudaUnregisterFatBinary", NULL, nodeId);
}

void dce___cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle), nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(hostFun), nodeId);
    CudaRtFrontend::AddStringForArguments(deviceFun, nodeId);
    CudaRtFrontend::AddStringForArguments(deviceName, nodeId);
    CudaRtFrontend::AddVariableForArguments(thread_limit, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(tid, 1, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(bid, 1, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(bDim, 1, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(gDim, 1, nodeId);
    CudaRtFrontend::AddHostPointerForArguments(wSize, 1, nodeId);

    CudaRtFrontend::Execute("cudaRegisterFunction", NULL, nodeId);

    deviceFun = CudaRtFrontend::GetOutputString(nodeId);
    tid = CudaRtFrontend::GetOutputHostPointer<uint3 > (1, nodeId);
    bid = CudaRtFrontend::GetOutputHostPointer<uint3 > (1, nodeId);
    bDim = CudaRtFrontend::GetOutputHostPointer<dim3 > (1, nodeId);
    gDim = CudaRtFrontend::GetOutputHostPointer<dim3 > (1, nodeId);
    wSize = CudaRtFrontend::GetOutputHostPointer<int>(1, nodeId);
}

void dce___cudaRegisterVar(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle), nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar), nodeId);
    CudaRtFrontend::AddStringForArguments(deviceAddress, nodeId);
    CudaRtFrontend::AddStringForArguments(deviceName, nodeId);
    CudaRtFrontend::AddVariableForArguments(ext, nodeId);
    CudaRtFrontend::AddVariableForArguments(size, nodeId);
    CudaRtFrontend::AddVariableForArguments(constant, nodeId);
    CudaRtFrontend::AddVariableForArguments(global, nodeId);
    CudaRtFrontend::Execute("cudaRegisterVar", NULL, nodeId);
}

void dce___cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle), nodeId);
    CudaRtFrontend::AddStringForArguments((char *) devicePtr, nodeId);
    CudaRtFrontend::Execute("cudaRegisterShared", NULL, nodeId);
}

void dce___cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
        size_t size, size_t alignment, int storage) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle), nodeId);
    CudaRtFrontend::AddStringForArguments((char *) devicePtr, nodeId);
    CudaRtFrontend::AddVariableForArguments(size, nodeId);
    CudaRtFrontend::AddVariableForArguments(alignment, nodeId);
    CudaRtFrontend::AddVariableForArguments(storage, nodeId);
    CudaRtFrontend::Execute("cudaRegisterSharedVar", NULL, nodeId);
}

void dce___cudaRegisterTexture(void **fatCubinHandle,
        const textureReference *hostVar, void **deviceAddress, char *deviceName,
        int dim, int norm, int ext) {
	uint32_t nodeId = UtilsGetNodeId ();
    CudaRtFrontend::Prepare(nodeId);
    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle), nodeId);
    // Achtung: passing the address and the content of the textureReference
    CudaRtFrontend::AddHostPointerForArguments(hostVar, 1, nodeId);
    CudaRtFrontend::AddVariableForArguments((uint64_t) hostVar, nodeId);
    CudaRtFrontend::AddStringForArguments((char *) deviceAddress, nodeId);
    CudaRtFrontend::AddStringForArguments(deviceName, nodeId);
    CudaRtFrontend::AddVariableForArguments(dim, nodeId);
    CudaRtFrontend::AddVariableForArguments(norm, nodeId);
    CudaRtFrontend::AddVariableForArguments(ext, nodeId);
    CudaRtFrontend::Execute("cudaRegisterTexture", NULL, nodeId);
}


/* */

int dce___cudaSynchronizeThreads(void** x, void* y) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement
    std::cerr << "*** Error: dce___cudaSynchronizeThreads() not yet implemented!"
            << std::endl;
    return 0;
}

void dce___cudaTextureFetch(const void *tex, void *index, int integer,
        void *val) {
	uint32_t nodeId = UtilsGetNodeId ();
    // FIXME: implement 
    std::cerr << "*** Error: dce___cudaTextureFetch() not yet implemented!" << std::endl;
}
