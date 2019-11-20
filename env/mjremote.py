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

import socket
import struct


class mjremote:

    nqpos = 0
    nmocap = 0
    ncamera = 0
    width = 0
    height = 0
    _s = None


    def _recvall(self, buffer):
        view = memoryview(buffer)
        while len(view):
            nrecv = self._s.recv_into(view)
            view = view[nrecv:]


    # result = (nqpos, nmocap, ncamra, width, height)
    def connect(self, address = '127.0.0.1', port = 1050):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.setsockopt(socket.SOL_TCP,  socket.TCP_NODELAY, 1)
        self._s.connect((address, port))
        data = bytearray(20)
        self._recvall(data)
        result = struct.unpack('iiiii', data)
        self.nqpos, self.nmocap, self.ncamera, self.width, self.height = result
        return result


    def close(self):
        if self._s:
            self._s.close()
            self._s = None


    # result = (key, active, select, refpos[3], refquat[4])
    def getinput(self):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 1))
        data = bytearray(40)
        self._recvall(data)
        result = struct.unpack('iiifffffff', data)
        return result


    # buffer = bytearray(3*width*height)
    def getimage(self, buffer):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 2))
        self._recvall(buffer)

    def getsegmentationimage(self, buffer):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 8))
        self._recvall(buffer)

    def getdepthimage(self, buffer):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 12))
        self._recvall(buffer)

    def savesnapshot(self):
        if not self._s:
            return 'Not connected'
        self._s.send(struct.pack("i", 3))

    def savevideoframe(self):
        if not self._s:
            return 'Not connected'
        self._s.send(struct.pack("i", 4))

    def setcamera(self, index):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 5))
        self._s.sendall(struct.pack("i", index))

    # qpos = numpy.ndarray(nqpos)
    def setqpos(self, qpos):
        if not self._s:
            return 'Not connected'
        if len(qpos)!=self.nqpos:
            return 'qpos has wrong size'
        fqpos = qpos.astype('float32')
        self._s.sendall(struct.pack("i", 6))
        self._s.sendall(fqpos.tobytes())

    def changeworld(self, string):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 9))
        self._s.sendall(struct.pack("i", len(string)))
        self._s.send(string.encode())

        self._s.sendall(struct.pack("i", 10))
        data = bytearray(20)
        self._recvall(data)
        result = struct.unpack('iiiii', data)
        self.nqpos, self.nmocap, self.ncamera, self.width, self.height = result
        return result

    # pos = numpy.ndarray(3*nmocap), quat = numpy.ndarray(4*nmocap)
    def setmocap(self, pos, quat):
        if not self._s:
            return 'Not connected'
        if len(pos)!=3*self.nmocap:
            return 'pos has wrong size'
        if len(quat)!=4*self.nmocap:
            return 'quat has wrong size'
        fpos = pos.astype('float32')
        fquat = quat.astype('float32')
        self._s.sendall(struct.pack("i", 7))
        self._s.sendall(fpos.tobytes())
        self._s.sendall(fquat.tobytes())

    def randomize_appearance(self):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 11))
        return


    # For Furniture Assembly Environment
    def setresolution(self, width, height):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 13))
        self._s.sendall(struct.pack("i", width))
        self._s.sendall(struct.pack("i", height))
        self.height = height
        self.width = width

    def getinput(self):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 15))
        data = bytearray(6)
        self._recvall(data)
        result = struct.unpack('6s', data)
        result = result[0].decode('utf-8').strip()
        return result

    def setgeompos(self, name, pos):
        if not self._s:
            return 'Not connected'
        fpos = pos.astype('float32')
        self._s.sendall(struct.pack("i", 14))
        self._s.sendall(fpos.tobytes())
        self._s.sendall(struct.pack("i", len(name)))
        self._s.send(name.encode())

    def setbackground(self, name):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 16))
        if name is None:
            name = ""

        self._s.sendall(struct.pack("i", len(name)))
        self._s.send(name.encode())

    def setquality(self, quality):
        if not self._s:
            return 'Not connected'
        self._s.sendall(struct.pack("i", 17))
        self._s.sendall(struct.pack("i", quality))

