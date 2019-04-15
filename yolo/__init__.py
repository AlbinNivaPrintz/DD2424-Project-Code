import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class YOLO(nn.Module):
    activation_functions = {
        "leaky": nn.LeakyReLU,
    }
    def __init__(self):
        self.optimizer = None
        self.channels = None
        self.layers = []
    
    @classmethod
    def from_config(clf, config):
        from configreader import Config
        net = clf()
        cfg = Config()
        cfg.read(config)
        for layer in cfg.iterate_layers():
            ltype = type(layer).__name__
            if ltype == "Net":
                # TODO Parse and understand all net settings
                net.channels = layer.config["channels"]
                # Call this last?
                # optimizer_cfg = {
                #     "momentum" = layer.config["momentum"],
                #     "learning_rate" = layer.config["learning_rate"]
                # }
            elif ltype == "Convolutional":
                if len(net.layers) > 0:
                    idx = net.__get_last_conv()
                    in_channels = net.layers[idx].out_channels
                else:
                    in_channels = net.channels
                # The actual convolution
                net.layers.append(
                    nn.Conv2d(
                        in_channels,
                        layer.config["filters"],
                        layer.config["size"],
                        stride=layer.config["stride"],
                        padding=layer.config["pad"])
                    )
                if layer.config["batch_normalize"]:
                    net.layers.append(
                        nn.BatchNorm2d(layer.config["filters"])
                    )
                net.layers.append(
                    net.activation_functions[layer.config["activation"]]
                )
            elif ltype == "Shortcut":
                # TODO
                raise KeyError(ltype + " isn't implemented yet!")
            elif ltype == "Yolo":
                # TODO
                raise KeyError(ltype + " isn't implemented yet!")
            elif ltype == "Route":
                # TODO
                raise KeyError(ltype + " isn't implemented yet!")
            elif ltype == "Upsample":
                # TODO
                raise KeyError(ltype + " isn't implemented yet!")
            else:
                raise KeyError(ltype + " isn't implemented!")

    def __get_layer_names(self):
        return [type(x).__name__ for x in self.layers]

    def __get_last_conv(self):
        names = self.__get_layer_names()
        print(names)
        return len(names) - 1 - names[::-1].index('Conv2d')
