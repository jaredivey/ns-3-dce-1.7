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

/*Binds an address as a texture reference.*/
CUresult dce_cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
	return cuTexRefSetArray(hTexRef, hArray, Flags);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hArray, nodeId);
//    CudaDrFrontend::AddVariableForArguments(Flags, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefSetArray", NULL, nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Sets the addressing mode for a texture reference.*/
CUresult dce_cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
	return cuTexRefSetAddressMode(hTexRef, dim, am);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(dim, nodeId);
//    CudaDrFrontend::AddVariableForArguments(am, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefSetAddressMode", NULL, nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Sets the filtering mode for a texture reference.*/
CUresult dce_cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
	return cuTexRefSetFilterMode(hTexRef, fm);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(fm, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefSetFilterMode", NULL, nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Sets the flags for a texture reference.*/
CUresult dce_cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
	return cuTexRefSetFlags(hTexRef, Flags);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(Flags, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefSetFlags", NULL, nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Sets the format for a texture reference.*/
CUresult dce_cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
	return cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddVariableForArguments(NumPackedComponents, nodeId);
//    CudaDrFrontend::AddVariableForArguments(fmt, nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefSetFormat", NULL, nodeId);
//    return (CUresult) (CudaDrFrontend::GetExitCode(nodeId));
}

/*Gets the address associated with a texture reference. */
CUresult dce_cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
	return cuTexRefGetAddress(pdptr, hTexRef);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefGetAddress", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pdptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Gets the array bound to a texture reference.*/
CUresult dce_cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
	return cuTexRefGetArray(phArray, hTexRef);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefGetArray", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *phArray = (CUarray) (CudaDrFrontend::GetOutputDevicePointer(nodeId));
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Gets the flags used by a texture reference. */
CUresult dce_cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
	return cuTexRefGetFlags(pFlags, hTexRef);
//	uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::Execute("cuTexRefGetFlags", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *pFlags = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

/*Binds an address as a texture reference.*/
CUresult dce_cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
	return cuTexRefSetAddress(ByteOffset, hTexRef, dptr, bytes);
// uint32_t nodeId = UtilsGetNodeId ();
//    CudaDrFrontend::Prepare(nodeId);
//    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef, nodeId);
//    CudaDrFrontend::AddVariableForArguments(dptr, nodeId);
//    CudaDrFrontend::AddVariableForArguments(bytes, nodeId);
//    CudaDrFrontend::Execute("cuTexRefSetAddress", NULL, nodeId);
//    if (CudaDrFrontend::Success(nodeId))
//        *ByteOffset = *(CudaDrFrontend::GetOutputHostPointer<size_t > (1, nodeId));
//    return (CUresult) CudaDrFrontend::GetExitCode(nodeId);
}

CUresult dce_cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefGetAddressMode() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefGetFilterMode() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefGetFormat() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefSetAddress2D() not yet implemented!" << endl;
    return (CUresult) 1;
}

CUresult dce_cuTexRefCreate(CUtexref *pTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefCreate() not yet implemented! : DEPRECATED" << endl;
    return (CUresult) 1;
}

CUresult dce_cuTexRefDestroy(CUtexref hTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefDestroy() not yet implemented! : DEPRECATED" << endl;
    return (CUresult) 1;
}
