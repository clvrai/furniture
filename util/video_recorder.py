import os

import moviepy.editor as mpy


class VideoRecorder(object):
    def __init__(self, video_dir='./'):
        self._frames = []

        self._video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)

    def reset(self):
        self._frames = []

    def add(self, frame):
        self._frames.append(frame * 255)

    def save_video(self, fname, fps=15.):
        path = os.path.join(self._video_dir, fname)

        def f(t):
            frame_length = len(self._frames)
            new_fps = 1./(1./fps + 1./frame_length)
            idx = min(int(t*new_fps), frame_length-1)
            return self._frames[idx]

        video = mpy.VideoClip(f, duration=len(self._frames)/fps+2)

        video.write_videofile(path, fps, verbose=False)
        self.reset()

