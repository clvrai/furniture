import triad_openvr
import time
import sys
import struct
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('10.0.1.48', 8051)

v = triad_openvr.triad_openvr()
v.print_discovered_objects()

if len(sys.argv) == 1:
    interval = 1/250
elif len(sys.argv) == 2:
    interval = 1/float(sys.argv[1])
else:
    print("Invalid number of arguments")
    interval = False
    
if interval:
    while(True):
        start = time.time()
        txt = ""
        data =  v.devices["tracker_1"].get_pose_quaternion()
        sent = sock.sendto(struct.pack('d'*len(data), *data), server_address)
        print("\r" + txt, end="")
        sleep_time = interval-(time.time()-start)
        if sleep_time>0:
            time.sleep(sleep_time)