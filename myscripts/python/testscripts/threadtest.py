import threading
import socket
import sys

total = 0
buffer_size = 1000

# called by each thread
def print_id(s, n, m):
    for ss in xrange(s,m,n):
        print "[ID " + str(s) + "] " + str(ss)

for i in xrange(1000):
    t = threading.Thread(target=print_id, args = (i, 500, 500000))
    t.daemon = True
    t.start()

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
