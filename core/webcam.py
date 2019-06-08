from threading import Thread
import cv2


class WebcamVideoStream:
    def __init__(self, src=1):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
 
    def read(self):
        return self.grabbed, self.frame
 
    def stop(self):
        self.stopped = True

    def get_width(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self):
        return int(self.stream.get(cv2.CAP_PROP_FPS))

    def is_open(self):
        return self.stream.isOpened()

    def set_frame_position(self, framePos):
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, framePos)

    def get_frame_position(self):
        return int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))

    def get_frame_count(self):
        return int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
