import socket
import sys
import ping

try:
    ping.verbose_ping('127.0.0.1', count=1000)
except socket.error, e:
    print "Ping Error:", e

total = 0
buffer_size = 1000

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
