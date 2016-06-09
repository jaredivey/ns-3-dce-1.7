import socket
import sys

total = 0
buffer_size = 1000

digits = 25000
scale = 10000
maxarr = 2000 + digits
arrinit = 2000
carry = 0
arr = [arrinit for i in range(digits+1)]

for i in xrange(digits, 1, -14):
    total = 0
    for j in xrange(i, 0, -1):
        total = (total * j) + (scale * arr[j])
        arr[j] = total % ((j * 2) - 1)
        total = total / ((j * 2) - 1)
    sys.stdout.write("%04d" % (carry + (total / scale)))
    carry = total % scale

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
