import yologru
from yologru import YoloGru
import yolo
import torch
import json
import argparse

def log(item):
    with open("training.log", "a") as f:
        f.write(str(item)+"\n")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--pretrained_model")
args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.pretrained_model is None:
    log("Loading network architecture from config.")
    net = YoloGru("conf/yolov3-gru.cfg")

    log("Loading weights.")
    net.load_weights("weights/yolov3.weights")
else:
    log("Loading model from "+args.pretrained_model)
    net = YoloGru.load(filename=args.pretrained_model)

net.to(torch.device("cuda:0"))

log("Loading data meta data file.")
with open("vtb_train.json", "r") as f:
    meta = json.load(f)

log("Begin training")
for i in range(args.epochs):
    import random
    random.shuffle(meta)
    for data_dict in meta:
        log("epoch {}, sequence {}".format(i+1, data_dict["name"]))
        # Should do training here 
        data_dir = "data/" + data_dict["name"]
        truth = yolo.data_from_path(
                data_dir + "/img",
                suffix=".jpg",
                bbfilename=data_dir+"/groundtruth_rect.txt",
                bbclass=data_dict["class"],
                bbsep=data_dict["sep"]
                )
        X_list = []
        X = truth["X"]
        X.cuda()
        for j in range(X.size(0)):
            X_list.append(X[j:j+1])
        data = list(zip(X_list, truth["labels"]))
        net.train_one_epoch(data)
    # Save one model per epoch (just to be safe)
    net.dump(filename="model_{}.pkl".format(i))

