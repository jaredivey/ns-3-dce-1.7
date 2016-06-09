import socket
import sys
import ping

if len (sys.argv) > 3:
    cur_address = sys.argv[1]
    cur_size = int(sys.argv[2])
    cur_count = int(sys.argv[3])
    try:
        ping.verbose_ping(cur_address, count=cur_count, packet_size=cur_size)
    except socket.error, e:
        print "Ping Error:", e

