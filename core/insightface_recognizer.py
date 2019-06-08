from mtcnn_insightface.recognizer.arcface_recognizer import Arcface_recognizer
import config
from object_detection.utils import visualization_utils as vis_util

recognizer = Arcface_recognizer(config.arc_model_name, config.arc_model_path, config.mtcnn_model_path)


def insight_recognize(image_np, frame):
    names, bounding_boxes = recognizer.recognize(image_np)

    if len(names) != 0:
        for idx, name in enumerate(names):
            if name:
                vis_util.draw_bounding_box_on_image_array(
                    frame, bounding_boxes[idx][1], bounding_boxes[idx][0],
                    bounding_boxes[idx][1] + bounding_boxes[idx][3],
                    bounding_boxes[idx][0] + bounding_boxes[idx][2],
                    color='Magenta', display_str_list=[str(name)],
                    use_normalized_coordinates=False)

    return frame