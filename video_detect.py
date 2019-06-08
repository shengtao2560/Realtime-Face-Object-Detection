import time

import cv2

import config
from core.aip_multisearch import aip_msearch
from core.tf_operation import load_graph, od_detect

cap = cv2.VideoCapture(config.input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(config.output_video_path, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),
                      (frame_width, frame_height))

sess, detection_graph = load_graph()

while True:
    start = time.time()
    ret, image_np = cap.read()
    if ret:
        frame = image_np.copy()
        frame = od_detect(sess, image_np, frame, detection_graph)
        frame = aip_msearch(image_np, frame, config.v_detect_group)

        out.write(frame)
        cv2.imshow('frame', frame)
        print(time.time() - start - 0.1)
        cv2.waitKey(int(1 / 30 * 1000 / 1))
        time.sleep(0.1)
    else:
        break

sess.close()
cap.release()
out.release()
cv2.destroyAllWindows()
