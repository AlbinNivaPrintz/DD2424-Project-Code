import yolo
from yolo import YOLO
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

        activation_kernels_i = torch.zeros(hidden_channels, self.in_channels, size, size)
        reset_kernels_i = torch.zeros(hidden_channels, self.in_channels, size, size)
        activation_kernels_h = torch.zeros(hidden_channels, self.in_channels, size, size)
        reset_kernels_h = torch.zeros(hidden_channels, self.in_channels, size, size)
        kernels_i = torch.zeros(hidden_channels, self.in_channels, size, size)
        half = size // 2
        for i in range(self.in_channels):
            kernels_i[i, i, half, half] = 1
        kernels_h = torch.zeros(hidden_channels, self.in_channels, size, size)

        std = 1 / (self.in_channels*size**2)
        # Random initalization
        torch.normal(
                activation_kernels_i,
                std*torch.ones_like(activation_kernels_i),
                out=activation_kernels_i
        )
        torch.normal(
                reset_kernels_i,
                std*torch.ones_like(activation_kernels_i),
                out=reset_kernels_i
        )
        torch.normal(
                activation_kernels_h,
                std*torch.ones_like(activation_kernels_i),
                out=activation_kernels_h
        )
        torch.normal(
                reset_kernels_i,
                std*torch.ones_like(activation_kernels_i),
                out=reset_kernels_h
        )
        torch.normal(
                kernels_i,
                std*torch.ones_like(kernels_i),
                out=kernels_i
        )
        torch.normal(
                kernels_h,
                std*torch.ones_like(kernels_h),
                out=kernels_h
        )
        self.activation_kernels_i = Parameter(activation_kernels_i)
        self.reset_kernels_i = Parameter(reset_kernels_i)
        self.activation_kernels_h = Parameter(activation_kernels_h)
        self.reset_kernels_h = Parameter(reset_kernels_h)
        self.kernels_i = Parameter(kernels_i)
        self.kernels_h = Parameter(kernels_h)

    
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        if hx is None:
            hx = input.clone().detach()
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
        
       
class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        
    def forward(self, labels, this_frame, last_frame=None):
        if last_frame is None:
            last_frame = this_frame
        frame_loss = torch.zeros(labels.size(0)-1)
        # Actual frame error
        frame_loss[:2] = (this_frame[:2] - labels[:2])**2
        frame_loss[2:4] = (torch.log(this_frame[2:4]/labels[2:4]))**2  # This is actually also squared error in t hehe
        frame_loss[4] = 
        # Smootheness error

        print(this_frame, labels)
        quit()
    

class YoloGru(YOLO):
    
    def __init__(self, config, labels="labels/coco.names"):
        super(YoloGru, self).__init__(config, labels, False)
        self.map_lfc_layer["ConvGru"] = self.__lfc_gru
        self.memory = {}
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
        block = nn.ModuleList([])
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
        
    def forward(self, x, gpu=False, keep=False):
        block_record = [x]
        output = None
        scale = None
        if keep:
            original = None
        for i, block in enumerate(self.layers):
            this_block = block_record[i]
            for j, mod in enumerate(block):
                if isinstance(mod, cr.Route):
                    this_block = self.forward_route(mod, this_block, block_record, i)
                elif isinstance(mod, cr.Shortcut):
                    this_block = self.forward_shortcut(mod, this_block, block_record, i)
                elif isinstance(mod, cr.Yolo):
                    # Detection layer
                    if keep:
                        formatted, non_format, num_scale = self.forward_yolo(mod, this_block, keep=True)
                        this_scale = torch.ones(non_format.size(1))*num_scale
                    else:
                        formatted = self.forward_yolo(mod, this_block)
                    if output is None:
                        output = formatted
                    else:
                        output = torch.cat(
                            (output, formatted),
                            dim=1
                        )
                    if keep:
                        if original is None:
                            original = non_format
                        else:
                            original = torch.cat(
                                (original, non_format),
                                dim=1
                            )
                        if scale is None:
                            scale = this_scale
                        else:
                            scale = torch.cat((scale, this_scale))
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
        if keep:
            return output, original, scale
        else:
            return output
        
    def train(self, data, parameters={"epochs": 2}):
        criterion = YoloLoss()
        optimizer = o.Adam(self.parameters())
        
        running_loss = 0.0
        for epoch in range(parameters["epochs"]):
            last_frame = None
            # zero parameter gradients
            for i, one_data in enumerate(data):
                
                X, label = one_data
                optimizer.zero_grad()
                # Transform labels to top_left bottom_right
                formatted_label = torch.zeros_like(label)
                formatted_label[0] = label[0]
                formatted_label[1] = label[1]
                formatted_label[2] = label[0] + label[2]
                formatted_label[3] = label[1] + label[3]
                formatted_label[4] = label[4]
                formatted_label[5] = label[5]
        
                # Want: tx ty bw bh of the best predictor
                # forward + backward + optimizer
                outputs, originals, scales = self(X, keep=True)
                originals = originals[0]  # t's
                bbs, idx = self.bbs_from_detection(outputs, 0.5, 0.5, return_idx=True) # b's
                bbs = bbs[0]
                originals = torch.cat([originals[i].unsqueeze(0) for i in idx], dim=0)
                # Here, we have the relevant ones!!!!
                # This function should find the one and only relevant thing.
                bb = self.most_similar_idx(formatted_label, bbs)
                # Find index of this bb
                idx = int(torch.prod(bbs == bb, 1).nonzero())
                original = originals[idx]
                scale = scales[idx]

                this_frame = bb.clone()
                this_frame[0:2] = original[0:2]
                bb[0] = bb[0] + (bb[2] - bb[0]) / 2
                bb[1] = bb[1] + (bb[3] - bb[1]) / 2
                c_x = bb[0]/scale - torch.sigmoid(original[0])
                c_y = bb[1]/scale - torch.sigmoid(original[1])
                formatted_label[0] += label[2]/2
                formatted_label[1] += label[3]/2
                formatted_label[0] /= scale
                formatted_label[1] /= scale
                bx = formatted_label[0] - c_x
                by = formatted_label[1] - c_y
                eps = torch.Tensor([1e-5])
                bx = min(1-eps, max(bx, eps))
                by = min(1-eps, max(by, eps))
                formatted_label[0] = torch.log((bx)/(1 - bx))
                formatted_label[1] = torch.log((by)/(1 - by))
                loss = criterion(formatted_label, this_frame, last_frame)
                loss.backward()
                optimizer.step()
        
                # print stats
                running_loss += loss.item()
                if i % 10 == 9:
                    print(running_loss/10)
                    running_loss = 0.0
                
        print("Done")

    def most_similar_idx(self, label, bbs):
        bbs = bbs[bbs[:, 4] == label[4], :]
        ious = self.iou(label, bbs)
        idx = torch.argmax(ious)
        return bbs[idx]

    def dump(self, filename="model.pkl"):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.layers, f)

    @classmethod
    def load(cls, config="conf/yolov3-gru.cfg", labels="labels/coco.names", filename="model.pkl"):
        import pickle
        with open(filename, 'rb') as f:
            layers = pickle.load(f)
        net = cls(config, labels)
        net.layers = layers
        net.train_yolo = False
        return net
        
