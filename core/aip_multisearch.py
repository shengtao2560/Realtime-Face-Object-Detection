import base64

import cv2

import config
from core.baidu_aip import AipFace
from object_detection.utils import visualization_utils as vis_util

client = AipFace(config.APP_ID, config.API_KEY, config.SECRET_KEY)


def aip_msearch(image_np, frame, groupId):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    imageType = "BASE64"
    options = {}
    options["max_face_num"] = 10
    options["max_user_num"] = 1
    res = client.multiSearch(image_code, imageType, groupId, options)
    # print(res)
    if res['result']:
        for face in res['result']['face_list']:
            x, y, z, w = face['location']['left'], face['location']['top'], face['location']['width'], \
                         face['location']['height']
            x, y, z, w = int(x), int(y), int(z), int(w)
            if face['user_list']:
                person_id = face['user_list'][0]['user_id']
                # print(person_id, x,y,z,w)
                if face['user_list'][0]['score'] > 80:
                    vis_util.draw_bounding_box_on_image_array(
                        frame, y, x, y + w, x + z, color='Magenta', display_str_list=[
                            person_id + ': ' + str(round(face['user_list'][0]['score'])) + '%'],
                        use_normalized_coordinates=False)
    return frame
