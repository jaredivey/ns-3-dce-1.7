import socket
import sys

if len (sys.argv) > 4:
    server = sys.argv[1]
    portno = int(sys.argv[2])
    buffer_size = int(sys.argv[3])
    loop_count = int(sys.argv[4])
    total = 0;

    try:
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = (server, portno)
        print >>sys.stderr, 'connecting to %s port %s' % server_address
        sock.connect(server_address)
        for i in range(loop_count):
            # Send data
            message = bytearray([1 for b in range(buffer_size) ])
            print >>sys.stderr, 'sending %d bytes' % len(message)
            sock.sendall(message)

            total += len(message)
            print >>sys.stderr, 'sent %d total bytes' % total

    finally:
        print >>sys.stderr, 'closing socket'
        sock.close()
