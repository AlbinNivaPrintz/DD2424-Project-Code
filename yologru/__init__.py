import yolo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o
import configreader as cr
import numpy as np
import cv2
from bidict import bidict


class ConvGru2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, size, key, pad=0, stride=1):
        from torch.nn import Parameter
        super(ConvGru2d, self).__init__()
        self.key = key
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.size = size
        self.pad = pad
        self.stride = stride
        self.activation_kernels_i = Parameter(
            torch.Tensor(hidden_channels, self.in_channels, size, size)
        )
        self.reset_kernels_i = Parameter(
            torch.Tensor(hidden_channels, self.in_channels, size, size)
        )
        self.activation_kernels_h = Parameter(
            torch.Tensor(hidden_channels, self.in_channels, size, size)
        )
        self.reset_kernels_h = Parameter(
            torch.Tensor(hidden_channels, self.in_channels, size, size)
        )
        self.kernels_i = Parameter(
            torch.Tensor(hidden_channels, self.in_channels, size, size)
        )
        self.kernels_h = Parameter(
            torch.Tensor(hidden_channels, self.in_channels, size, size)
        )
    
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_channels, input.size(2), input.size(3))
        return self.conv_gru(
            input, hx,
            self.activation_kernels_i, self.reset_kernels_i, self.kernels_i,
            self.activation_kernels_h, self.reset_kernels_h, self.kernels_h
        )

    def conv_gru(
            self, input, hx,
            activation_kernels_i, reset_kernels_i, kernels_i,
            activation_kernels_h, reset_kernels_h, kernels_h
        ):
        # Activation
        zi = F.conv2d(input, activation_kernels_i, padding=self.pad, stride=self.stride)
        zh = F.conv2d(hx, activation_kernels_h, padding=self.pad, stride=self.stride)
        z = torch.sigmoid(zi + zh)
        # Reset
        ri = F.conv2d(input, reset_kernels_i, padding=self.pad, stride=self.stride)
        rh = F.conv2d(hx, reset_kernels_h, padding=self.pad, stride=self.stride)
        r = torch.sigmoid(ri + rh)
        # Candidate
        hi = F.conv2d(input, kernels_i, padding=self.pad, stride=self.stride)
        hh = F.conv2d(r*hx, kernels_h, padding=self.pad, stride=self.stride)
        h_tilde = torch.tanh(hi + hh)
        h_new = (1 - z)*hx + z*h_tilde
        return h_new
    

class YoloGru(yolo.YOLO):
    
    def __init__(self, config, labels="labels/coco.names"):
        super(YoloGru, self).__init__(config, init_layers=False)
        self.map_lfc_layer["ConvGru"] = self.__lfc_gru
        self.memory = {}
        self._train_yolo = False
        self._memory_key_counter = 0
        self.layers_from_config(config)
        self._train_yolo = False
        
    @property
    def train_yolo(self):
        return self._train_yolo
    
    @train_yolo.setter
    def train_yolo(self, train_yolo):
        self._train_yolo = train_yolo
        for block in self.layers:
            for mod in block:
                if isinstance(mod, nn.Module) and not isinstance(mod, ConvGru2d):
                    mod.requires_grad = train_yolo
        
    def __lfc_gru(self, layer, outfilters, current_channels):
        # Parse a gru layer and insert it into the self.layers
        block = []
        # This should (if needed) be a ConvGru2d later
        block.append(ConvGru2d(
            current_channels,
            layer.config["filters"],
            size=layer.config["size"],
            key=self._memory_key_counter,
            pad=layer.config["pad"],
            stride=layer.config["stride"]
        ))
        self.memory[self._memory_key_counter] = None
        self._memory_key_counter += 1
        return {
            "channels": current_channels,
            "outfilters": layer.config["filters"],
            "block": block
        }
        
    def forward(self, x, gpu=False):
        block_record = [x]
        output = None
        for i, block in enumerate(self.layers):
            this_block = block_record[i]
            for j, mod in enumerate(block):
                if isinstance(mod, cr.Route):
                    this_block = self.forward_route(mod, this_block, block_record, i)
                elif isinstance(mod, cr.Shortcut):
                    this_block = self.forward_shortcut(mod, this_block, block_record, i)
                elif isinstance(mod, cr.Yolo):
                    # Detection layer
                    formatted = self.forward_yolo(mod, this_block)
                    if output is None:
                        output = formatted
                    else:
                        output = torch.cat(
                            (output, formatted),
                            dim=1
                        )
                elif isinstance(mod, cr.Upsample):
                    this_block = self.forward_upsample(mod, this_block)
                elif isinstance(mod, nn.Module):
                    try:
                        if isinstance(mod, ConvGru2d):
                            this_block = mod(this_block, self.memory[mod.key])
                            self.memory[mod.key] = this_block
                        else:
                            this_block = mod(this_block)
                    except RuntimeError as e:
                        print(["{}\n".format(x.size(1)) for x in block_record])
                        raise e
            block_record.append(this_block)
        return output
        
    def train(X, Y):
        optimizer = o.Adam()
        
