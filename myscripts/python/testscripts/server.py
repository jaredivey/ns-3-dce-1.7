import socket
import sys

if len (sys.argv) > 1:
    portno = int(sys.argv[1])
    total = 0;

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
                while True:
                    data = connection.recv(1024)
                    if data:
                        total += len(data)
                        print >>sys.stderr, 'received %d bytes' % total
                    else:
                        break

            finally:
                # Clean up the connection
                connection.close()
    finally:
        sock.close()
