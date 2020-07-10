import distutils.spawn
import distutils.version
import glob
import os
import os.path
import subprocess

import moviepy.editor as mpy
import numpy as np


class VideoRecorder(object):
    """
    Records videos; Either stores all frames in RAM or writes to file at every frame.
    Choose depending on performance or RAM size.
    """

    def __init__(
        self,
        record_mode="RAM",  # ram or file
        video_dir="./videos",
        prefix="default",
        demo_dir=None,
        frames_per_sec=15,
        output_frames_per_sec=15,
    ):
        self._video_dir = video_dir
        self._record_mode = record_mode
        os.makedirs(video_dir, exist_ok=True)
        self._demo_dir = demo_dir
        self.set_outfile(prefix)
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec
        self.encoder = None
        self._frames = []

    def set_outfile(self, prefix):
        if self._demo_dir:
            # make video filename same as corresponding demo filename
            count = min(9999, self._get_demo_count(prefix, self._demo_dir))
        else:
            # make seperate video only filename
            prefix = prefix + "vidonly"
            count = min(9999, self._get_vid_count(prefix))
        video_name = prefix + "{:04d}.mp4".format(count)
        self.outfile = os.path.join(self._video_dir, video_name)

    def _get_demo_count(self, prefix, demo_dir):
        return len(glob.glob(os.path.join(demo_dir, prefix) + "*"))

    def _get_vid_count(self, prefix):
        return len(glob.glob(os.path.join(self._video_dir, prefix) + "*"))

    def capture_frame(self, frame, render_mode="from_env_render"):
        """``
        Render the given `env` and add the resulting frame to the video.
        render_mode: from_env_render or rgb_array
        if data is from env.render(), we will need to convert it to 0-255
        if data is rgb array, then no need to convert
        """
        if self._record_mode == "RAM":
            if render_mode == "rgb_array":
                self._frames.append(frame)
            elif render_mode == "from_env_render":
                self._frames.append(255 * frame)
        elif self._record_mode == "file":
            if render_mode == "rgb_array":
                self._encode_image_frame(frame)
            elif render_mode == "from_env_render":
                self._encode_image_frame((255 * frame).astype("uint8"))

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(
                self.outfile,
                frame.shape,
                self.frames_per_sec,
                self.output_frames_per_sec,
            )
        try:
            self.encoder.capture_frame(frame)
        except Exception:
            pass
            # logger.warn todo

    def close(self, name=None, success=True):
        """
        Closes the video file, and optionally renames it.
        Make sure to manually close, or else you'll leak the encoder process
        """
        if name is not None:
            self.outfile = os.path.join(self._video_dir, name)

        if self._record_mode == "RAM":
            if success:
                fps = self.output_frames_per_sec

                def f(t):
                    frame_length = len(self._frames)
                    new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
                    idx = min(int(t * new_fps), frame_length - 1)
                    return self._frames[idx]

                video = mpy.VideoClip(f, duration=len(self._frames) / fps + 2)
                video.write_videofile(self.outfile, fps, verbose=False)
                self._frames = []

        elif self._record_mode == "file":
            if self.encoder is not None:
                self.encoder.close()
                self.encoder = None
            else:
                # No frames captured. Set metadata, and remove the empty output file.
                os.remove(self.outfile)
    
        if success:
            print("closed vr, video at", self.outfile)

class ImageEncoder(object):
    def __init__(self, outfile, frame_shape, frames_per_sec, output_frames_per_sec):
        self.proc = None
        self.outfile = outfile
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise Exception(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e., RGB values for a w-by-h image, with an optional alpha channel.".format(
                    frame_shape
                )
            )
        self.wh = (w, h)
        self.includes_alpha = pixfmt == 4
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.output_frames_per_sec = output_frames_per_sec

        if distutils.spawn.find_executable("ffmpeg") is not None:
            self.backend = "ffmpeg"
        else:
            raise ImportError(
                "Could not find ffmpeg executable. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it."
            )
        print("starting")
        self.start()

    def start(self):
        self.cmdline = (
            self.backend,
            "-nostats",
            "-loglevel",
            "error",  # suppress warnings
            "-y",
            # input
            "-f",
            "rawvideo",
            "-s:v",
            "{}x{}".format(*self.wh),
            "-pix_fmt",
            ("rgb32" if self.includes_alpha else "rgb24"),
            "-framerate",
            "%d" % self.frames_per_sec,
            "-i",
            "-",  # this used to be /dev/stdin, which is not Windows-friendly
            # output
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "%d" % self.output_frames_per_sec,
            self.outfile,
        )

        print('Starting ffmpeg with "%s"', " ".join(self.cmdline))
        if hasattr(os, "setsid"):  # setsid not present on Windows
            self.proc = subprocess.Popen(
                self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid
            )
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            print("type error")
            raise TypeError(
                "Wrong type {} for {} (must be np.ndarray or np.generic)".format(
                    type(frame), frame
                )
            )
        if frame.shape != self.frame_shape:
            print("shape error")
            raise Exception(
                "Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(
                    frame.shape, self.frame_shape
                )
            )
        if frame.dtype != np.uint8:
            print(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(
                    frame.dtype
                )
            )
            raise TypeError(
                "Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(
                    frame.dtype
                )
            )
        if distutils.version.LooseVersion(
            np.__version__
        ) >= distutils.version.LooseVersion("1.9.0"):
            # print('writing')
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            print("VideoRecorder encoder exited with status {}".format(ret))
