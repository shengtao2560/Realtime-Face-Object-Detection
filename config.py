# -*- coding: utf-8 -*-
import os

# TensorFlow Object Detection
PATH_TO_CKPT = 'object_detection/ssd_mobilenet_v2_coco/frozen_inference_graph.pb'
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# recognizer switch
face_detect = 'baidu'

# BaiDu_AI
APP_ID = '16280566'
API_KEY = 'EGeAu91Ci8W29YxwYGj1YOVg'
SECRET_KEY = '14ksadHT3wF3FMQBlqamCMXSu42pof15'

# MTCNN & insightface
# model parameters
model_params = {"backbone_type": "resnet_v2_m_50",
                "out_type": "E",
                "bn_decay": 0.9,
                "weight_decay": 0.0005,
                "keep_prob": 0.4,
                "embd_size": 512}

# evaluation parameters
eval_dropout_flag = False
eval_bn_flag = False

# face database parameters
custom_dir = '../data/custom'
arc_model_name = 'Arcface-330000'
arc_model_path = './model/Arcface_model/Arcface-330000'

base_dir = './model/MTCNN_model'
mtcnn_model_path = [os.path.join(base_dir, "Pnet_model/Pnet_model.ckpt-20000"),
                    os.path.join(base_dir, "Rnet_model/Rnet_model.ckpt-40000"),
                    os.path.join(base_dir, "Onet_model/Onet_model.ckpt-40000")]
embds_save_dir = "../data/face_db"

vs_src = 1
# vs_src = 'rtmp://www.youyizhongxue.cn/live/detect'

r_pool_size = 1
r_queue_size = 5
r_detect_group = 'student'

# 视频检测
input_video_path = './input/i_demo2.mp4'
output_video_path = './output/out8.mp4'
v_detect_group = 'test'

# 多线程视频检测
m_pool_size = 20
m_queue_size = 150
