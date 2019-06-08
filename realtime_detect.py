import time
from multiprocessing import Queue, Pool

import config
from core.detect import *


def realtime():
    input_q = Queue(maxsize=config.r_queue_size)
    output_q = Queue(maxsize=config.r_queue_size)
    pool = Pool(config.r_pool_size, worker, (input_q, output_q))
    vs = WebcamVideoStream(config.vs_src).start()

    count_frame = 0
    last_frame = time.time()
    while True:
        ret, frame = vs.read()
        count_frame = count_frame + 1
        if ret:
            input_q.put(frame)
            output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", output_rgb)
            print(round(time.time() - last_frame, 3))
            last_frame = time.time()
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.terminate()
    vs.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realtime()
