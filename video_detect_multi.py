from multiprocessing import Queue, Pool
from queue import PriorityQueue

import config
from core.detect import *


def video():
    input_q = Queue(maxsize=config.m_queue_size)
    output_q = Queue(maxsize=config.m_queue_size)
    output_pq = PriorityQueue(maxsize=3 * config.m_queue_size)
    pool = Pool(config.m_pool_size, worker, (input_q, output_q))

    vs = cv2.VideoCapture(config.input_video_path)

    out = cv2.VideoWriter(config.output_video_path, cv2.VideoWriter_fourcc(*'XVID'), vs.get(cv2.CAP_PROP_FPS),
                          (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    countReadFrame = 0
    countWriteFrame = 1
    nFrame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    firstReadFrame = True
    firstTreatedFrame = True
    firstUsedFrame = True
    while True:
        if not input_q.full():
            ret, frame = vs.read()
            if ret:
                input_q.put((int(vs.get(cv2.CAP_PROP_POS_FRAMES)), frame))
                countReadFrame = countReadFrame + 1
                if firstReadFrame:
                    print(" --> Reading first frames from input file. Feeding input queue.\n")
                    firstReadFrame = False

        if not output_q.empty():
            output_pq.put(output_q.get())
            if firstTreatedFrame:
                print(" --> Recovering the first treated frame.\n")
                firstTreatedFrame = False

        if not output_pq.empty():
            prior, output_frame = output_pq.get()
            if prior > countWriteFrame:
                output_pq.put((prior, output_frame))
            else:
                countWriteFrame = countWriteFrame + 1
                output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                out.write(output_rgb)
                cv2.imshow('frame', output_rgb)

                if firstUsedFrame:
                    print(" --> Start using recovered frame (displaying and/or writing).\n")
                    firstUsedFrame = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("Read frames: %-3i %% -- Write frame: %-3i %%" % (
            int(countReadFrame / nFrame * 100), int(countWriteFrame / nFrame * 100)), end='\r')
        if (not ret) & input_q.empty() & output_q.empty() & output_pq.empty():
            break

    print(
        "\nFile have been successfully read and treated:\n  --> {}/{} read frames \n  --> {}/{} write frames \n".format(
            countReadFrame, nFrame, countWriteFrame - 1, nFrame))

    pool.terminate()
    vs.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video()
