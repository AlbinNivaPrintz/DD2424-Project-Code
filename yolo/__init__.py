import torch
import torch.nn as nn
import torch.nn.functional as F
import configreader as cr
import numpy as np
import cv2
from bidict import bidict


class YOLO(nn.Module):
    activation_functions = {
        "leaky": nn.LeakyReLU,
    }

    def __init__(self, config, labels="labels/coco.names", init_layers=True):
        super(YOLO, self).__init__()
        self.labelmap = self.get_label_map(labels)
        self.optimizer = None
        self.channels = 0
        self.n_classes = 0
        self.layers = []
        self.im_size = (0, 0)
        self.map_lfc_layer = {
            "Net": self.__lfc_net,
            "Convolutional": self.__lfc_convolutional,
            "Shortcut": self.__lfc_shortcut,
            "Yolo": self.__lfc_yolo,
            "Upsample": self.__lfc_upsample,
            "Route": self.__lfc_route
        }
        if init_layers:
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
                if isinstance(mod, nn.Module):
                    mod.requires_grad = train_yolo

    @classmethod
    def from_config(cls, config, labels="labels/coco.names"):
        net = cls(labels)
        net.layers_from_config(config)
        return net
        
    def get_label_map(self, filename):
    	onewaymap = {}
    	with open(filename, 'r') as f:
    		l = f.readlines()
    	for i in range(len(l)):
    		onewaymap[i] = l[i].strip()
    	return bidict(onewaymap)

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
            try:
                out = self.map_lfc_layer[ltype](layer, outfilters, current_channels)
            except KeyError:
                raise KeyError(
                    ltype + " was found in config, but isn't an implemented block!"
                )
            if "channels" in out:
                current_channels = out["channels"]
            if "block" in out:
                self.layers.append(out["block"])
                outfilters.append(out["outfilters"])
            assert len(outfilters) == len(self.layers)
            
    def __lfc_net(self, layer, outfilters, current_channels):
        self.channels = layer.config["channels"]
        self.im_size = (layer.config["width"], layer.config["height"])
        return {"channels": self.channels}
        
    def __lfc_convolutional(self, layer, outfilters, current_channels):
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
        out = {
            "channels": layer.config["filters"],
            "outfilters": layer.config["filters"],
            "block": block
        }
        return out
    
    def __lfc_shortcut(self, layer, outfilters, current_channels):
        return {"outfilters": current_channels, "block": [layer]}
        
    def __lfc_yolo(self, layer, outfilters, current_channels):
        if self.n_classes != layer.config["classes"] and self.n_classes != 0:
            # Weird config thing with classes being detection layer specific
            print("There are different numbers of classes in the yolo layers.")
        self.n_classes = layer.config["classes"]
        return {"outfilters": 0, "block": [layer]}
    
    def __lfc_route(self, layer, outfilters, current_channels):
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
        return {"channels": c1+c2, "outfilters": c1+c2, "block": [layer]}
    
    def __lfc_upsample(self, layer, outfilters, current_channels):
        return {"outfilters": current_channels, "block": [layer]}

    def __get_layer_names(self):
        unraveled = []
        for y in self.layers:
            for x in y:
                unraveled.append(type(x).__name__)
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
                        this_block = mod(this_block)
                    except RuntimeError as e:
                        print(["{}\n".format(x.size(1)) for x in block_record])
                        raise e
            block_record.append(this_block)
        return output
        
    def forward_shortcut(self, mod, this_block, block_record, i):
        return this_block + block_record[i+mod.config["from"]+1]
        
    def forward_route(self, mod, this_block, block_record, i):
        if mod.config["layers"][0] < 0:
            idx = i + mod.config["layers"][0] + 1
        else:
            idx = mod.config["layers"][0]
        if len(mod.config["layers"]) == 1:
            return block_record[idx]
        else:
            if mod.config["layers"][1] < 0:
                idx2 = i + mod.config["layers"][1] + 1
            else:
                idx2 = mod.config["layers"][1]
            return torch.cat((block_record[idx], block_record[idx2]), dim=1)
        
    def forward_upsample(self, mod, this_block):
        return F.interpolate(
            this_block,
            scale_factor=mod.config["stride"], mode="bilinear", align_corners=False
        )
        
    def forward_yolo(self, mod, this_block):
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
        return formatted
        
    def detect(self, X, outfilename="test.png", gpu=False):
        print("Performing forward pass.")
        with torch.nograd():
            out = self.forward(X, gpu)
        print("Calculating bounding boxes.")
        bbs = net.bbs_from_detection(out, 0.5, 0.5)
        print("Drawing boxes.")
        img_draw = net.draw_bbs(img, bbs[0])
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test.png", img_draw)
        print("Done! Wrote to {}.".format(outfilename))
                

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
            non_zero = bounding_box[i, bounding_box[i, :, 5]!=0, :]
            # TODO Here we are gonna do NMS
            unique_classes = torch.unique(non_zero[:, 4]).type(torch.IntTensor)
            output_this_frame = None
            for cls in unique_classes:
                # Perform NMS class wise
                cls = int(cls)
                of_this_class = non_zero[non_zero[:, 4]==cls, :]
                order = torch.argsort(of_this_class[:, 5], descending=True)
                of_this_class = of_this_class[order, :]
                for i in range(of_this_class.size(0)):
                    current_box = of_this_class[i, :]
                    if current_box[5] == 0: continue
                    other_boxes = of_this_class[i+1:, :]
                    scores = self.iou(current_box, other_boxes)
                    passed = scores < nms_threshold
                    of_this_class[i+1:, :] *= passed.type(torch.FloatTensor).unsqueeze(1)
                if output_this_frame is None:
                    output_this_frame = of_this_class[of_this_class[:, 5]!=0, :]
                else:
                    output_this_frame = torch.cat(
                        (output_this_frame, of_this_class[of_this_class[:, 5]!=0, :]),
                        dim=0)
            output.append(output_this_frame)
        return output
        
    def iou(self, bbox, bboxes):
        x1, y1, x2, y2, _, _ = bbox
        area = (x2 - x1)*(y2 - y1)
        areas = (bboxes[:, 2] - bboxes[:, 0])*(bboxes[:, 3] - bboxes[:, 1])
        inter_x1 = torch.where(bboxes[:, 0] > x1, bboxes[:, 0], x1)
        inter_y1 = torch.where(bboxes[:, 1] > y1, bboxes[:, 1], y1)
        inter_x2 = torch.where(bboxes[:, 2] < x2, bboxes[:, 2], x2)
        inter_y2 = torch.where(bboxes[:, 3] < y2, bboxes[:, 3], y2)
        unsigned_inter = (inter_x2 - inter_x1)*(inter_y2 - inter_y1)
        inter_areas = torch.where(unsigned_inter > 0, unsigned_inter, torch.zeros_like(unsigned_inter))
        union = area + areas - inter_areas
        return inter_areas/union

    def draw_bbs(self, x, bbs):
        # TODO This probably draw stuff in a more clever way. Text placements and stuff.
        img_draw = x.copy()
        true_w, true_h, _ = x.shape
        wfactor = true_h / self.im_size[0]
        hfactor = true_w / self.im_size[1]
        for r in range(bbs.size(0)):
            x1, y1, x2, y2, c, p = bbs[r]
            topleft = (int(wfactor*x1), int(hfactor*y1))
            bottomright = (int(wfactor*x2), int(hfactor*y2))
            c = int(c)
            img_draw = cv2.rectangle(img_draw, topleft, bottomright, (255, 0, 0), 2)
            bottomleft_text = (topleft[0]+10, topleft[1]+40)
            img_draw = cv2.putText(
            	img_draw,
            	self.labelmap[c],
            	bottomleft_text, 
            	cv2.FONT_HERSHEY_PLAIN,
            	4, 
            	(255, 0, 0),
            	4
            )
        return img_draw
        
    def detect_on_webcam(self, size=416):
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            x, im = cv_to_torch(frame, size=size)
            out = self.forward(x, gpu=False)
            bbs = self.bbs_from_detection(out, 0.5, 0.5)
            if bbs[0] is not None:
                img_draw = self.draw_bbs(im, bbs[0])
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                # Display the resulting frame
                cv2.imshow('detection',img_draw)
            else:
                cv2.imshow('detection',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        
    def detect_on_video(self, filename, size=416):
        cap = cv2.VideoCapture(filename)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            x, im = cv_to_torch(frame, size=size)
            out = self.forward(x, gpu=False)
            bbs = self.bbs_from_detection(out, 0.5, 0.5)
            if bbs[0] is not None:
                img_draw = self.draw_bbs(im, bbs[0])
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                # Display the resulting frame
                cv2.imshow('detection',img_draw)
            else:
                cv2.imshow('detection',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()


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
        im_out = cv2.resize(im_rgb, (size, size), cv2.INTER_AREA)
    else:
        size = im_rgb.shape[0]
    try:
        return torch.Tensor(im_out.transpose((2, 0, 1)).reshape((1, 3, size, size))), im_rgb
    except ValueError as e:
        print("Images must be square!")
        raise e
    # When everything done, release the capture
    cap.release()