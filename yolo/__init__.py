import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # def load_weights(self, weightfile):
    #     # Open the weights file
    #     fp = open(weightfile, "rb")
    #
    #     # The first 5 values are header information
    #     # 1. Major version number
    #     # 2. Minor Version Number
    #     # 3. Subversion number
    #     # 4,5. Images seen by the network (during training)
    #     header = np.fromfile(fp, dtype=np.int32, count=5)
    #     self.header = torch.from_numpy(header)
    #     self.seen = self.header[3]
    #
    #     weights = np.fromfile(fp, dtype=np.float32)
    #
    #     ptr = 0
    #     for i in range(len(self.layers)):
    #         module_type = type(self.layers[i][0])
    #
    #         # If module_type is convolutional load weights
    #         # Otherwise ignore.
    #
    #         if module_type == nn.Conv2d:
    #             model = self.layers[i]
    #             try:
    #                 batch_normalize = isinstance(self.layers[i][1], nn.BatchNorm2d)
    #             except:
    #                 batch_normalize = False
    #
    #             conv = model[0]
    #
    #             if batch_normalize:
    #                 bn = model[1]
    #
    #                 # Get the number of weights of Batch Norm Layer
    #                 num_bn_biases = bn.bias.numel()
    #
    #                 # Load the weights
    #                 bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
    #                 ptr += num_bn_biases
    #
    #                 bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
    #                 ptr += num_bn_biases
    #
    #                 bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
    #                 ptr += num_bn_biases
    #
    #                 bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
    #                 ptr += num_bn_biases
    #
    #                 # Cast the loaded weights into dims of model weights.
    #                 bn_biases = bn_biases.view_as(bn.bias.data)
    #                 bn_weights = bn_weights.view_as(bn.weight.data)
    #                 bn_running_mean = bn_running_mean.view_as(bn.running_mean)
    #                 bn_running_var = bn_running_var.view_as(bn.running_var)
    #
    #                 # Copy the data to model
    #                 bn.bias.data.copy_(bn_biases)
    #                 bn.weight.data.copy_(bn_weights)
    #                 bn.running_mean.copy_(bn_running_mean)
    #                 bn.running_var.copy_(bn_running_var)
    #
    #             else:
    #                 # Number of biases
    #                 num_biases = conv.bias.numel()
    #
    #                 # Load the weights
    #                 conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
    #                 ptr = ptr + num_biases
    #
    #                 # reshape the loaded weights according to the dims of the model weights
    #                 conv_biases = conv_biases.view_as(conv.bias.data)
    #
    #                 # Finally copy the data
    #                 conv.bias.data.copy_(conv_biases)
    #
    #             # Let us load the weights for the Convolutional layers
    #             num_weights = conv.weight.numel()
    #
    #             # Do the same as above for weights
    #             conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
    #             ptr = ptr + num_weights
    #
    #             conv_weights = conv_weights.view_as(conv.weight.data)
    #             conv.weight.data.copy_(conv_weights)

    def load_weights(self,weightfile):

        #open the weights file
        fp = open(weightfile, "rb")
        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor version number
        # 3. Subversion number
        # 4,5 Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        header = torch.from_numpy(header)
        seen = header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        # To keep track of where we are in the weights array we initiate a position tracker ptr
        ptr = 0

        # module_list = layers
        # model = block
        for block in self.layers:

            if isinstance(block[0], nn.Conv2d):
                try:
                    batch_normalize = isinstance(block[1], nn.BatchNorm2d)
                except IndexError:
                    batch_normalize = False
                conv = block[0]

                if batch_normalize:
                    #save it as bn
                    try:
                        bn = block[1]
                    except IndexError:
                        break
                    # Get the number of weights of batch norm layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # "Cast the loaded weights into dims of model weights."
                    # Changing the shapes of the variables into "wanted shapes".
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
    #
    #                # Copy the data to the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Assign the number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases

                    # Reshape the loaded weights according to the dimensions of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Copy the data
                    conv.bias.data.copy_(conv_biases)

                # Load the weights for the convolutional layers
                # by doing the same as in the else-statement for the weights
                # number of biases
                num_weights = conv.weight.numel()
                # Load weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights
                # Reshape the loaded weights and then copy the data
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                
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
                    c_x_y = np.hstack((c_y, c_x)).repeat(len(anchors), axis=0)
                    c_x_y = torch.from_numpy(c_x_y).view(
                        (n_batch, grid_size*grid_size*len(anchors), 2)
                    )
                    formatted[:, :, 0:2] = torch.sigmoid(formatted[:, :, 0:2]) + c_x_y.float()
                    formatted[:, :, 4] = torch.sigmoid(formatted[:, :, 4])
                    formatted[:, :, 2:4] = expanded_anchors * torch.exp(formatted[:, :, 2:4])
                    formatted[:, :, 5:] = torch.sigmoid(formatted[:, :, 5:])
                    formatted[:, :, :4] *= scale
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

    def bbs_from_detection(self, detection, threshold, nms_threshold):
        # Suppress where objectness is lower than threshold
        mask = detection[:, :, 4] > threshold
        mask = mask.view((mask.size(0), -1, 1)).expand_as(detection).type(torch.FloatTensor)
        cleared = detection * mask

        # Get class prediction
        class_score, class_predictions = torch.max(cleared[:, :, 5:], 2)
        # Ensure the correct shape
        class_predictions = class_predictions.view((class_predictions.size(0), -1))

        # Transform to top_l_x, top_l_y, bottom_r_x, bottom_r_y
        bounding_box = torch.zeros_like(cleared[:, :, :6])
        bounding_box[:, :, 0] = cleared[:, :, 0] - cleared[:, :, 2] / 2
        bounding_box[:, :, 1] = cleared[:, :, 1] - cleared[:, :, 3] / 2
        bounding_box[:, :, 2] = cleared[:, :, 0] + cleared[:, :, 2] / 2
        bounding_box[:, :, 3] = cleared[:, :, 1] + cleared[:, :, 3] / 2
        bounding_box[:, :, 4] = class_predictions
        bounding_box[:, :, 5] = class_score

        output = []
        for i in range(bounding_box.size(0)):
            non_zero = bounding_box[i, cleared[i, :, 4] != 0, :]
            # TODO Here we are gonna do NMS
            output.append(non_zero)
        return output

    def draw_bbs(self, x, bbs):
        img_draw = np.array(x).copy().transpose((1, 2, 0))
        for r in range(bbs.size(0)):
            x1, y1, x2, y2, c, p = bbs[r]
            # Class stuff is correct, but location seems to be off......
            print((int(x1), int(y1)), (int(x2), int(y2)), int(c), float(p))
            img_draw = cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        return img_draw


def data_from_image(filename, size=416):
    im_bgr = cv2.imread(filename)
    return cv_to_torch(im_bgr, size=size)


def data_from_video(filename, size=416):
    cap = cv2.VideoCapture(filename)
    out = None
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            # End of video
            break
        tens = cv_to_torch(frame_bgr, size=size)
        if out is None:
            out = tens
        else:
            out = torch.cat((out, tens), dim=0)
        cv2.waitKey(0)
    cap.release()
    return out


def data_from_path(path, size=416):
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".png")]
    order = np.argsort([int(x.split(".")[0]) for x in images])
    images = [images[i] for i in order]
    out = None
    for image in images:
        im = cv2.imread(join(path, image))
        tens = cv_to_torch(im, size=size)
        if out is None:
            out = tens
        else:
            out = torch.cat((out, tens), dim=0)
    return out


def cv_to_torch(cv_img, size=None):
    im_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    if size is not None:
        im_rgb = cv2.resize(im_rgb, (size, size), cv2.INTER_AREA)
    else:
        size = im_rgb.shape[0]
    try:
        return torch.Tensor(im_rgb.transpose((2, 0, 1)).reshape((1, 3, size, size)))
    except ValueError as e:
        print("Images must be square!")
        raise e
    # When everything done, release the capture
    cap.release()