import os
import subprocess
import tempfile
import os.path
import distutils.spawn, distutils.version
import numpy as np
from six import StringIO
import six

class VideoRecorder(object):
    def __init__(self, video_dir='./', frames_per_sec=15, output_frames_per_sec=15):
        self._video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)
        self.video_name = 'default.mp4' # TODO, find vars to fix name
        self.outfile = os.path.join(self._video_dir, self.video_name)
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec
        self.encoder = None

    def capture_frame(self, frame, render_mode='rgb_array'):
        """Render the given `env` and add the resulting frame to the video."""
        # logger.debug('Capturing video frame: path=%s', self.path)
        render_mode = 'rgb_array' # change to allow many render types
        if render_mode == 'rgb_array':
            self._encode_image_frame(frame)


    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(self.outfile, frame.shape, self.frames_per_sec, self.output_frames_per_sec)
        try:
            self.encoder.capture_frame(frame)
        except Exception as e:
            pass
            # logger.warn todo

    def close(self):
        """Make sure to manually close, or else you'll leak the encoder process"""
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None
            print('closed vr')
        else:
            # No frames captured. Set metadata, and remove the empty output file.
            os.remove(self.outfile) 


class ImageEncoder(object):
    def __init__(self, outfile, frame_shape, frames_per_sec, output_frames_per_sec):
        self.proc = None
        self.outfile = outfile
        # Frame shape should be lines-first, so w and h are swapped
        print('frame_shape', frame_shape)
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame("Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., RGB values for a w-by-h image, with an optional alpha channel.".format(frame_shape))
        self.wh = (w,h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec
   
        if distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise ImportError('Could not find ffmpeg executable. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it.') 
        self.start()

    def start(self):
        self.cmdline = (self.backend,
                     '-nostats',
                     '-loglevel', 'error', # suppress warnings
                     '-y',

                     # input
                     '-f', 'rawvideo',
                     '-s:v', '{}x{}'.format(*self.wh),
                     '-pix_fmt',('rgb32' if self.includes_alpha else 'rgb24'),
                     '-framerate', '%d' % self.frames_per_sec,
                     '-i', '-', # this used to be /dev/stdin, which is not Windows-friendly

                     # output
                     '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                     '-vcodec', 'libx264',
                     '-pix_fmt', 'yuv420p',
                     '-r', '%d' % self.output_frames_per_sec,
                     self.outfile
                     )

        print('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os,'setsid'): #setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)


    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise TypeError('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise ShapeError("Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise TypeError("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())


    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            print("VideoRecorder encoder exited with status {}".format(ret))

