import numpy as np
import torch
import yolo
import configreader as cf


net = YOLO("conf/yolov3.cfg")
net.load_weights("yoloV3.weights")