import yolo
from yolo import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import configreader as cr
import numpy as np
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
    def __init__(self, params={"lr":1e-3, "len_seq": 20, "l_coord": 5, "l_noobj": 0.5}):
        super(YoloLoss, self).__init__()
        self.l_coord = params["l_coord"]
        self.l_noobj = params["l_noobj"]
        
    def forward(self, labels, this_frame, last_frame=None):
        if last_frame is None:
            last_frame = this_frame
        # Actual frame error
        frame_loss = torch.zeros_like(labels)
        ## For the ones with nan in 0 or 1 should only count objectiveness so make them equal prediction everywhere else
        nan_in_zero = torch.isnan(labels[:, 0])
        nan_in_one = torch.isnan(labels[:, 1])
        nan_in_any = nan_in_one + nan_in_zero > 0
        labels[nan_in_any, :4] = this_frame[nan_in_any, :4]
        labels[nan_in_any, 4] = -10
        labels[nan_in_any, 5:] = this_frame[nan_in_any, 5:]
        ## Last minute transform to probabilities of objectness and class probabilities
        labels[:, 4:] = torch.sigmoid(labels[:, 4:])
        this_frame[:, 4:] = torch.sigmoid(this_frame[:, 4:]) 
        ## Squared error on the t's
        frame_loss[:, :2] = (labels[:, :2] - this_frame[:, :2])**2
        frame_loss[:, 2:4] = (torch.log(labels[:, 2:4] / this_frame[:, 2:4]))**2
        ## Everything else gets binary cross entropy loss
        frame_loss[:, 4:] = labels[:, 4:]*torch.log(labels[:, 4:]/this_frame[:, 4:]) + (1-labels[:, 4:])*torch.log((1-labels[:, 4:])/(1-this_frame[:, 4:]))
        frame_loss[:, :4] *= self.l_coord
        frame_loss[nan_in_any, 4] *= self.l_noobj
        frame_loss_sum = torch.sum(frame_loss)

        # Smootheness error
        smoothness_loss_sum = 0

        return frame_loss_sum + smoothness_loss_sum
    

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
        c_x_y = None
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
                        formatted, non_format, num_scale, c_x_y_this = self.forward_yolo(mod, this_block, keep=True)
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
                        if c_x_y is None:
                            c_x_y = c_x_y_this[0]
                        else:
                            c_x_y = torch.cat((c_x_y, c_x_y_this[0]))
                elif isinstance(mod, cr.Upsample):
                    this_block = self.forward_upsample(mod, this_block)
                elif isinstance(mod, nn.Module):
                    try:
                        if isinstance(mod, ConvGru2d):
                            this_block = mod(this_block, self.memory[mod.key])
                            self.memory[mod.key] = this_block.detach()
                        else:
                            this_block = mod(this_block)
                    except RuntimeError as e:
                        print(["{}\n".format(x.size(1)) for x in block_record])
                        raise e
            block_record.append(this_block)
        if keep:
            return output, original, scale, c_x_y
        else:
            return output
        
    def train_one_epoch(self, data, optimizer, params={"len_seq": 20, "l_coord": 5, "l_noobj": 0.5}):
        import random
        criterion = YoloLoss(params)
        optimizer.zero_grad()
        
        running_loss = 0.0

        batched = [data[i:i + params["len_seq"]] for i in range(0, len(data), params["len_seq"])]
        random.shuffle(batched)

        for batch in batched:
            last_frame = None
            batch_loss = 0
            # zero parameter gradients
            for i, one_data in enumerate(batch):
                
                X, label = one_data
                # Want: tx ty bw bh of the best predictor
                # forward + backward + optimizer
                outputs, originals, scales, c_x_y = self(X, keep=True)
                originals = originals[0]  # t's
                # Transform labels to top_left bottom_right
                formatted_label = torch.zeros_like(originals)
                formatted_label[:, 0] = label[0] + label[2] / 2
                formatted_label[:, 1] = label[1] + label[3] / 2
                formatted_label[:, 2] = label[2]
                formatted_label[:, 3] = label[3]
                formatted_label[:, :4] /= scales.unsqueeze(1)
                formatted_label[:, :2] -= c_x_y
                formatted_label[:, :2] = torch.log((formatted_label[:, :2])/(1 - formatted_label[:, :2]))
                # These are going to be sigmoided later
                formatted_label[:, 4] = 10
                formatted_label[:, 5:] = torch.where(
                        torch.eye(originals.size(1)-5)[int(label[4])] == 1,
                        torch.ones(originals.size(1)-5)*10,
                        -torch.ones(originals.size(1)-5)*10)

                # Get t to b in case of width
                originals[:, 2:4] = outputs[0, :, 2:4]
                loss = criterion(formatted_label, originals, last_frame)
                batch_loss += loss

                # Detach since do not want to calculate gradients through here
                last_frame = originals.clone().detach()
                for key in self.memory:
                    self.memory[key].detach_()
        
                # print stats and update
                running_loss += loss.item()
                if i % 10 == 9:
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss = 0
                    optimizer.zero_grad()
                    log(running_loss/10)
                    running_loss = 0.0
            if batch_loss != 0:
                # Perhaps missed some since only every tenth
                batch_loss.backward()
                optimizer.step()
                batch_loss = 0
                optimizer.zero_grad()
                
        log("Done")

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
        
def log(item):
    with open("training.log", "a") as f:
        f.write(str(item)+"\n")
