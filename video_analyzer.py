import cv2
import numpy as np

class VideoAnalyzer:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.pixels_per_meter = 1.0
        self.origin = (0, 0) # (x, y) in pixels
        # Frame cache for faster scrubbing (stores last N frames)
        self._frame_cache = {}
        self._cache_order = []
        self._cache_max_size = 30

    def load_video(self, path):
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Clear frame cache for new video
        self._frame_cache = {}
        self._cache_order = []

    def get_frame(self, frame_idx):
        if self.cap is None:
            return None

        # Check cache first
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx].copy()

        # Read from video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Add to cache
            self._frame_cache[frame_idx] = rgb_frame.copy()
            self._cache_order.append(frame_idx)

            # Evict old frames if cache is full
            while len(self._cache_order) > self._cache_max_size:
                old_idx = self._cache_order.pop(0)
                if old_idx in self._frame_cache:
                    del self._frame_cache[old_idx]

            return rgb_frame
        return None

    def set_calibration(self, p1, p2, real_distance_m):
        """
        p1, p2: tuples of (x, y) pixels
        real_distance_m: float, meters
        """
        dist_pixels = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if dist_pixels == 0:
            raise ValueError("Calibration points cannot be the same")
        self.pixels_per_meter = dist_pixels / real_distance_m

    def set_origin(self, p):
        self.origin = p
