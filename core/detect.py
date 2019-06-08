from core.aip_multisearch import aip_msearch
from core.tf_operation import load_graph, od_detect
from core.webcam import *
from core.insightface_recognizer import insight_recognize
import config


def detect_objects(image_np, sess, detection_graph):
    frame = image_np.copy()
    frame = od_detect(sess, image_np, frame, detection_graph)

    if config.face_detect == 'baidu':
        frame = aip_msearch(image_np, frame, config.r_detect_group)
    elif config.face_detect == 'insight':
        frame = insight_recognize(image_np, frame)

    return frame


def worker(input_q, output_q):
    sess, detection_graph = load_graph()
    while True:
        frame = input_q.get()
        if len(frame) == 2:
            frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            output_q.put((frame[0], detect_objects(frame_rgb, sess, detection_graph)))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_q.put(detect_objects(frame_rgb, sess, detection_graph))
    sess.close()
