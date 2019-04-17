import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import configreader as cr


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
                    n_batch = this_block.size(0)
                    scale = self.im_size[0] // this_block.size(2)
                    grid_size = this_block.size(2)
                    label_dim = 5 + self.n_classes
                    anchors = [mod.config["anchors"][i] for i in mod.config["mask"]]

                    if output is None:
                        output = this_block.view((n_batch, grid_size*grid_size, len(anchors)*label_dim))
                    else:
                        output = torch.cat(
                            (output, this_block.view((n_batch, grid_size*grid_size, len(anchors)*label_dim))),
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
