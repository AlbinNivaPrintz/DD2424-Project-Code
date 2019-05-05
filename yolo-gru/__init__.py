import torch
import torch.nn as nn
import torch.nn.functional as F
import configreader as cr
import numpy as np
import cv2
from bidict import bidict


class Yolo-Gru(yolo.YOLO):
    
    def __init__(self):
        super(Yolo-Gru, self).__init__
        self.map_lfc_layer["gru"] = self.__lfc_gru
        
    def __lfc_gru(self, layer, outfilters, current_channels):
        # Parse a gru layer and insert it into the self.layers
        pass
        
