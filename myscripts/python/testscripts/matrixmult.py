import socket
import sys

total = 0
buffer_size = 1000

matrix_array = [[(i*buffer_size + j) for j in range(buffer_size)] for i in range(buffer_size)]

matrix_prod = [[sum(a*b for a,b in zip(i,j)) for j in zip(*matrix_array)] for i in matrix_array]

print matrix_prod

if len (sys.argv) > 1:
    portno = int(sys.argv[1])
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ("localhost", portno)
    print >>sys.stderr, 'connecting to %s port %s' % server_address
    sock.connect(server_address)

    try:
        # Send data
        message = bytearray([1 for b in range(buffer_size) ])
        print >>sys.stderr, 'sending %d bytes' % len(message)
        sock.sendall(message)

        total += len(message)
        print >>sys.stderr, 'sent %d total bytes' % total

    finally:
        print >>sys.stderr, 'closing socket'
        sock.close()
