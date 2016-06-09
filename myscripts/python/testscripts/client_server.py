import socket
import sys

if len (sys.argv) > 2:
    server = sys.argv[1]
    portno = int(sys.argv[2])
    total_sent = 0;
    total_recv = 0;

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port
    server_address = ('', portno)
    print >>sys.stderr, 'starting up on %s port %s' % server_address
    sock.bind(server_address)

    # Listen for incoming connections
    sock.listen(1)

    try:

        while True:
            # Wait for a connection
            print >>sys.stderr, 'waiting for a connection'
            connection, client_address = sock.accept()

            try:
                print >>sys.stderr, 'connection from', client_address

                # Receive the data in small chunks and retransmit it
                pkt_total = 0
                while True:
                    data = connection.recv(1024)
                    if data:
                        pkt_total += len(data)
                        total_recv += len(data)
                        print >>sys.stderr, 'received %d bytes' % total_recv
                    else:
                        break


                # Create a TCP/IP socket
                sendsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                # Connect the socket to the port where the server is listening
                server_address = (server, portno+1)
                print >>sys.stderr, 'connecting to %s port %s' % server_address
                sendsock.connect(server_address)

                try:
                    # Send data
                    message = bytearray([1 for b in range(pkt_total) ])
                    sendsock.sendall(message)

                    total_sent += len(message)
                    print >>sys.stderr, 'sent %d bytes' % total_sent

                finally:
                    print >>sys.stderr, 'closing socket'
                    sendsock.close()

            finally:
                # Clean up the connection
                connection.close()
    finally:
        sock.close()


        
