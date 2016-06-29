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

/**
 * @file   TcpCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 8 12:08:33 2009
 *
 * @brief
 *
 *
 */

#include "dce-tap-communicator.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;

int DceTapCommunicator::tap_value = 0;
short DceTapCommunicator::port_value = 50;

DceTapCommunicator::DceTapCommunicator(const std::string& communicator)
{
    mPort = port_value;
    std::ostringstream oss;
    oss << "128.0.0." << tap_value + 1;
    mHostname = oss.str();
    struct hostent *ent = gethostbyname(mHostname.c_str());
    if (ent == NULL)
        throw "DceTapCommunicator: Can't resolve hostname '" + mHostname + "'.";
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
}

DceTapCommunicator::DceTapCommunicator(const char *hostname, short port)
{
    std::ostringstream oss;
    oss << "128.0.0." << tap_value + 1;
    mHostname = oss.str();
    struct hostent *ent = gethostbyname(mHostname.c_str());
    if (ent == NULL)
        throw "DceTapCommunicator: Can't resolve hostname '" + mHostname + "'.";
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
    mPort = port_value;
}

DceTapCommunicator::DceTapCommunicator(int fd, const char *hostname)
{
    mSocketFd = fd;
    InitializeStream();
}

DceTapCommunicator::~DceTapCommunicator()
{
    delete[] mInAddr;
}

void DceTapCommunicator::Serve()
{
    struct sockaddr_in socket_addr;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw "DceTapCommunicator: Can't create socket.";

    memset((char *) &socket_addr, 0, sizeof(struct sockaddr_in));

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(port_value);
    socket_addr.sin_addr.s_addr = INADDR_ANY;

    char on = 1;
    setsockopt(mSocketFd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof (on));

    int result = bind(mSocketFd, (struct sockaddr *) & socket_addr,
            sizeof (struct sockaddr_in));
     if( result != 0)
        throw "DceTapCommunicator: Can't bind socket.";
     ++tap_value;

    if (listen(mSocketFd, 5) != 0)
        throw "DceTapCommunicator: Can't listen from socket.";
}

const Communicator * const DceTapCommunicator::Accept() const
{
    unsigned client_socket_fd;
    struct sockaddr_in client_socket_addr;

    unsigned client_socket_addr_size;

    client_socket_addr_size = sizeof (struct sockaddr_in);
    if ((client_socket_fd = accept(mSocketFd,
            (sockaddr *) & client_socket_addr,
            &client_socket_addr_size)) == 0)
        throw "DceTapCommunicator: Error while accepting connection.";

//    // Proactively create next tap for next simulated node
//    // Not needed for DCE side
//    char tapname[7];
//    sprintf (tapname, "tap%d", tap_value);
//    std::ostringstream oss;
//    oss << "128.0.0." << tap_value + 1;
//    CreateTap (tapname, "", oss.str().c_str(), "", "1", "255.255.255.0");

    return new DceTapCommunicator(client_socket_fd,
            inet_ntoa(client_socket_addr.sin_addr));
}

void DceTapCommunicator::Connect()
{
    struct sockaddr_in remote;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw "DceTapCommunicator: Can't create socket.";

    remote.sin_family = AF_INET;
    remote.sin_port = htons(port_value);
    memcpy(&remote.sin_addr, mInAddr, mInAddrSize);

    if (connect(mSocketFd, (struct sockaddr *) & remote,
            sizeof (struct sockaddr_in)) != 0)
        throw "DceTapCommunicator: Can't connect to socket.";
    InitializeStream();
    ++tap_value;
}

void DceTapCommunicator::Close()
{
}

size_t DceTapCommunicator::Read(char* buffer, size_t size)
{
    mpInput->read(buffer, size);
    if (mpInput->bad() || mpInput->eof())
        return 0;
    return size;
}

size_t DceTapCommunicator::Write(const char* buffer, size_t size)
{
    mpOutput->write(buffer, size);
    return size;
}

void DceTapCommunicator::Sync()
{
    mpOutput->flush();
}

void DceTapCommunicator::InitializeStream()
{
	mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::in);
	mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::out);
	mpInput = new istream(mpInputBuf);
	mpOutput = new ostream(mpOutputBuf);
}

