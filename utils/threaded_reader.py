import cv2
import threading
import queue
import time

class ThreadedVideoReader:
    def __init__(self, path, queue_size=256, start_frame=0):
        self.stream = cv2.VideoCapture(path)
        if start_frame > 0:
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.stopped = False
        self.queue = queue.Queue(maxsize=queue_size)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        
        # Start the reading thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stopped = True
                    return
                self.queue.put(frame)
            else:
                # Wait a tiny bit if queue is full to prevent CPU burning
                time.sleep(0.01)

    def read(self):
        # Return next frame in the queue
        return self.queue.get()

    def more(self):
        # Returns True if there are frames left in the Queue
        return not self.queue.empty() or not self.stopped

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()
