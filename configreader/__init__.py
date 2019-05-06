import re
from torch import nn

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
    
    def load_attr(self, attr):
        raise NotImplementedError

class Net(Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.name = "Net"
        self.config = {}

    def load_attr(self, attr):
        types = {
            "momentum": float,
            "batch": int,
            "subdivisions": int,
            "width": int,
            "height": int,
            "channels": int,
            "decay": float,
            "angle": float,
            "saturation": float,
            "exposure": float,
            "hue": float,
            "learning_rate": float,
            "burn_in": int,
            "max_batches": int,
            "policy": str,
            "steps": int,
            "scales": float,
        }
        attr = attr.replace(" ", "").split("=")
        if attr[0] in ["steps", "scales"]:
            values = attr[1].split(",")
            self.config[attr[0]] = tuple([ types[attr[0]](x) for x in values ])
        else:
            self.config[attr[0]] = types[attr[0]](attr[1])

class Convolutional(Layer):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.name = "Convolutional"
        self.config = {}

    def load_attr(self, attr):
        types = {
            "batch_normalize": int,
            "filters": int,
            "size": int,
            "stride": int,
            "pad": int,
            "activation": str,
        }
        attr = attr.replace(" ", "").split("=")
        self.config[attr[0]] = types[attr[0]](attr[1])

class Shortcut(Layer):
    def __init__(self):
        super(Shortcut, self).__init__()
        self.name = "Shortcut"
        self.config = {}

    def load_attr(self, attr):
        types = {
            "from": int,
            "activation": str,
        }
        attr = attr.replace(" ", "").split("=")
        self.config[attr[0]] = types[attr[0]](attr[1])

class Yolo(Layer):
    def __init__(self):
        super(Yolo, self).__init__()
        self.name = "Yolo"
        self.config = {}

    def load_attr(self, attr):
        types = {
            "mask": int,
            "anchors": int,
            "classes": int,
            "num": int,
            "jitter": float,
            "ignore_thresh": float,
            "truth_thresh": int,
            "random": int
        }
        attr = attr.replace(" ", "").split("=")
        if attr[0] in ["mask"]:
            values = attr[1].split(",")
            self.config[attr[0]] = tuple([ types[attr[0]](x) for x in values ])
        elif attr[0] in ["anchors"]:
            values = re.findall(r"\,".join([r"[^\,]+"] * 2), attr[1])
            self.config[attr[0]] = [ tuple([ types[attr[0]](x) for x in y.split(",") ]) for y in values ]
        else:
            self.config[attr[0]] = types[attr[0]](attr[1])

class Route(Layer):
    def __init__(self):
        super(Route, self).__init__()
        self.name = "Route"
        self.config = {}

    def load_attr(self, attr):
        types = {
            "layers": int,
        }
        attr = attr.replace(" ", "").split("=")
        if attr[0] in ["layers"]:
            values = attr[1].split(",")
            self.config[attr[0]] = tuple([ types[attr[0]](x) for x in values ])
        else:
            self.config[attr[0]] = types[attr[0]](attr[1])

class Upsample(Layer):
    def __init__(self):
        super(Upsample, self).__init__()
        self.name = "Upsample"
        self.config = {}

    def load_attr(self, attr):
        types = {
            "stride": int
        }
        attr = attr.replace(" ", "").split("=")
        self.config[attr[0]] = types[attr[0]](attr[1])
        
class ConvGru(Layer):
    def __init__(self):
        super(ConvGru, self).__init__()
        self.name = "ConvGru"
        self.config = {}
        
    def load_attr(self, attr):
        types = {
            "filters": int,
            "size": int,
            "stride": int,
            "pad": int,
        }
        attr = attr.replace(" ", "").split("=")
        self.config[attr[0]] = types[attr[0]](attr[1])


class Config:
    def __init__(self):
        self.layers = []

    def iterate_layers(self):
        for layer in self.layers:
            yield layer

    def __parse(self, filename):
        with open(filename, "r") as f:
            config = f.read()
        config = re.split(r"(?=\[[\w\s]+\]\n)", config)
        for block in config:
            block_list = re.split("\n", block)
            match = re.match(r"\[([\w\s]+)\]", block_list[0])
            if match is None:
                continue
            layer = self.__layer_from_name(match.group(1))()
            for line in self.__block_generator(block_list[1:]):
                layer.load_attr(line)
            self.layers.append(layer)
#        print(self.layers[0].config)
        
    def read(self, filename):
        self.__parse(filename)
        
    def __block_generator(self, iterable):
        for x in iterable:
            if x not in [""] and not x.startswith("#"):
                yield x

    def __layer_from_name(self, name):
        layers = {
            "": None,
            "net": Net,
            "convolutional": Convolutional,
            "shortcut": Shortcut,
            "yolo": Yolo,
            "route": Route,
            "upsample": Upsample,
            "convgru": ConvGru,
        }
        return layers[name]

