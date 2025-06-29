import cv2
import threading

class Gelsight:
    """
    Continuously captures frames from a camera in a background thread.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        """
        Return the most recent frame.
        """
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        """
        Stop the background thread and release the camera.
        """
        self.stopped = True
        self.cap.release()

# Module-level singleton to minimize latency
_stream = None


def init_gelsight(src=0):
    """
    Initialize the global Gelsight stream if not already started.
    Returns the Gelsight instance.
    """
    global _stream
    if _stream is None:
        _stream = Gelsight(src).start()
    return _stream


def get_latest_frame(src=0):
    """
    Quickly return the latest frame from the Gelsight camera.
    Initializes the stream on first call.
    Returns (grabbed: bool, frame: ndarray|None).
    """
    stream = init_gelsight(src)
    return stream.read()
