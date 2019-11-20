'''
Original work Copyright 2019 Roboti LLC
Modified work Copyright 2019 Panasonic Beta, a division of Panasonic Corporation of North America

Redistribution and use of this file (hereafter "Software") in source and 
binary forms, with or without modification, are permitted provided that 
the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
'''

from mjremote import mjremote
import time
import socket
import numpy as np
from matplotlib import pyplot as plt

remote_host =  socket.gethostbyname("localhost")
m = mjremote()
#print('Connect: ', m.connect())
print('Connect: ',  m.connect(remote_host))



print('Request new world: ', m.changeworld("./test/latest/world/1551848929_lever_blue_floatinghook.xml"))

exit()

c = bytearray(3*m.width*m.height)
s = bytearray(3*m.width*m.height)

now = time.time()

m.getimage(c)
m.getsegmentationimage(s)

done = time.time()

m.randomize_appearance()

print("got 100 images in {} seconds ({} images/second)".format(done-now, 100/(done-now)))


try:
    nds = np.flipud(np.frombuffer(s, dtype=np.uint8).reshape(m.height,m.width,3))
    ndc = np.flipud(np.frombuffer(c, dtype=np.uint8).reshape(m.height,m.width,3))
    
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(ndc)
    fig.add_subplot(1, 2, 2)
    plt.imshow(nds)
    #plt.imshow(nds[:,:,0], cmap=plt.cm.Paired)
    plt.show()

    print("Show plot")

except Exception as e: 
    print(e)

m.close()
