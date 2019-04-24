import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configreader as cr
import numpy as np
import cv2


class YOLO(nn.Module):
    activation_functions = {
        "leaky": nn.LeakyReLU,
    }

    def __init__(self):
        super(YOLO, self).__init__()
        self.optimizer = None
        self.channels = 0
        self.n_classes = 0
        self.layers = []
        self.im_size = (0, 0)

    @classmethod
    def from_config(cls, config):
        net = cls()
        net.layers_from_config(config)
        return net

    def layers_from_config(self, config):
        outfilters = []
        current_channels = 0
        cfg = cr.Config()
        cfg.read(config)
        for layer in cfg.iterate_layers():
            ltype = type(layer).__name__
            if ltype == "Net":
                # TODO Parse and understand all net settings
                self.channels = layer.config["channels"]
                self.im_size = (layer.config["width"], layer.config["height"])
                current_channels = self.channels
                # Call this last?
                # optimizer_cfg = {
                #     "momentum" = layer.config["momentum"],
                #     "learning_rate" = layer.config["learning_rate"]
                # }
            elif ltype == "Convolutional":
                block = []
                pad = (layer.config["size"] - 1) // 2 if layer.config["pad"] else 0
                block.append(
                    nn.Conv2d(
                        current_channels,
                        layer.config["filters"],
                        layer.config["size"],
                        stride=layer.config["stride"],
                        padding=pad)
                    )
                if "batch_normalize" in layer.config and layer.config["batch_normalize"]:
                    block.append(
                        nn.BatchNorm2d(layer.config["filters"])
                    )
                if layer.config["activation"] != "linear":
                    block.append(
                        self.activation_functions[layer.config["activation"]]()
                    )
                current_channels = layer.config["filters"]
                outfilters.append(layer.config["filters"])
                self.layers.append(block)
            elif ltype == "Shortcut":
                outfilters.append(current_channels)
                self.layers.append([layer])
            elif ltype == "Yolo":
                outfilters.append(0)
                # Weird config thing with classes being detection layer specific
                if self.n_classes != layer.config["classes"] and self.n_classes != 0:
                    print("There are different numbers of classes in the yolo layers. Just so you know.")
                self.n_classes = layer.config["classes"]
                self.layers.append([layer])
            elif ltype == "Route":
                if layer.config["layers"][0] < 0:
                    c1 = outfilters[len(self.layers) + layer.config["layers"][0]]
                else:
                    c1 = outfilters[layer.config["layers"][0]]
                try:
                    if layer.config["layers"][1] < 0:
                        c2 = outfilters[len(self.layers) + layer.config["layers"][1]]
                    else:
                        c2 = outfilters[layer.config["layers"][1]]
                except IndexError:
                    c2 = 0
                current_channels = c1+c2
                outfilters.append(c1+c2)
                self.layers.append([layer])
            elif ltype == "Upsample":
                self.layers.append([layer])
                outfilters.append(current_channels)
            else:
                raise KeyError(ltype + " isn't implemented!")
            assert len(outfilters) == len(self.layers)

    def __get_layer_names(self):
        unraveled = []
        for y in self.layers:
            for x in y:
                unraveled.append(type(x).__name__)
        print(unraveled)
        return unraveled

    def __get_last_conv(self):
        names = self.__get_layer_names()
        return len(names) - 1 - names[::-1].index('Conv2d')

    def forward(self, x, gpu):
        block_record = [x]
        output = None
        for i, block in enumerate(self.layers):
            this_block = block_record[i]
            for j, mod in enumerate(block):
                if isinstance(mod, cr.Route):
                    if mod.config["layers"][0] < 0:
                        idx = i + mod.config["layers"][0] + 1
                    else:
                        idx = mod.config["layers"][0]
                    if len(mod.config["layers"]) == 1:
                        this_block = block_record[idx]
                    else:
                        if mod.config["layers"][1] < 0:
                            idx2 = i + mod.config["layers"][1] + 1
                        else:
                            idx2 = mod.config["layers"][1]
                        this_block = torch.cat((block_record[idx], block_record[idx2]), dim=1)
                elif isinstance(mod, cr.Shortcut):
                    this_block = this_block + block_record[i+mod.config["from"]+1]
                elif isinstance(mod, cr.Yolo):
                    # Detection layer
                    n_batch = this_block.size(0)
                    scale = self.im_size[0] // this_block.size(2)
                    grid_size = this_block.size(2)
                    label_dim = 5 + self.n_classes
                    anchors = [mod.config["anchors"][i] for i in mod.config["mask"]]
                    # Make anchors relative to grid
                    anchors = torch.tensor([[anchor[0]/scale, anchor[1]/scale] for anchor in anchors])
                    expanded_anchors = anchors.repeat(1, grid_size * grid_size, 1)
                    # Desired format (bx, by, bw, bh, c, pc1, ..., pcC) of length label_dim
                    formatted = this_block.view((n_batch, len(anchors)*label_dim, grid_size*grid_size))
                    formatted = formatted.transpose(1, 2).contiguous()
                    formatted = formatted.view(
                        (n_batch, grid_size*grid_size*len(anchors), label_dim)
                    )
                    # Transform t_x to b_x
                    c_x = np.repeat(np.arange(0, grid_size), grid_size).reshape((-1, 1))
                    c_y = np.tile(np.arange(0, grid_size), grid_size).reshape((-1, 1))
                    c_x_y = np.hstack((c_x, c_y)).repeat(len(anchors), axis=0)
                    c_x_y = torch.from_numpy(c_x_y).view(
                        (n_batch, grid_size*grid_size*len(anchors), 2)
                    )
                    formatted[:, :, 0:2] = torch.sigmoid(formatted[:, :, 0:2]) + c_x_y.float()
                    formatted[:, :, 4] = torch.sigmoid(formatted[:, :, 4])
                    formatted[:, :, 2:4] = expanded_anchors * torch.exp(formatted[:, :, 2:4])
                    formatted[:, :, 5:] = torch.sigmoid(formatted[:, :, 5:])
                    formatted[:, :, :4] = formatted[:, :, :4]
                    if output is None:
                        output = formatted
                    else:
                        output = torch.cat(
                            (output, formatted),
                            dim=1
                        )
                elif isinstance(mod, cr.Upsample):
                    this_block = F.interpolate(
                        this_block,
                        scale_factor=mod.config["stride"], mode="bilinear", align_corners=False
                    )
                elif isinstance(mod, nn.Module):
                    try:
                        this_block = mod(this_block)
                    except RuntimeError as e:
                        print(["{}\n".format(x.size(1)) for x in block_record])
                        raise e
            block_record.append(this_block)
        return output

    @staticmethod
    def data_from_image(filename, size = 416):
        im_bgr = cv2.imread(filename)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(im_rgb, (size, size), cv2.INTER_AREA)
        im = im_resized.transpose((2, 0, 1)).reshape((1, 3, size, size))
        return torch.Tensor(im)