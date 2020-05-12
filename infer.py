import argparse
import os
import sys
import numpy
import cv2

import torch

import models
import utils
import exporters

def parse_arguments(args):
    usage_text = (
        "Semi-supervised Spherical Depth Estimation Testing."        
    )
    parser = argparse.ArgumentParser(description=usage_text)    
    parser.add_argument("--input_path", type=str, help="Path to the input spherical panorama image.")
    parser.add_argument('--weights', type=str, help='Path to the trained weights file.')    
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')        
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    # device & visualizers
    device = torch.device("cuda:{}" .format(gpus[0])\
        if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0\
        else "cpu")    
    # model    
    model = models.get_model("resnet_coord", {})
    utils.init.initialize_weights(model, args.weights, pred_bias=None)
    model = model.to(device)
    # test data
    width, height = 512, 256
    if not os.path.exists(args.input_path):
        print("Input image path does not exist (%s)." % args.input_path)
        exit(-1)
    img = cv2.imread(args.input_path)
    h, w, _ = img.shape
    if h != height and w != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = img.transpose(2, 0, 1) / 255.0
    img = torch.from_numpy(img).float().expand(1, -1, -1, -1)
    model.eval()
    with torch.no_grad():                    
        left_rgb = img.to(device)
        ''' Prediction '''
        left_depth_pred = torch.abs(model(left_rgb))
        exporters.image.save_data(os.path.join(
            os.path.dirname(args.input_path),
            os.path.splitext(os.path.basename(
                args.input_path))[0] + "_depth.exr"),
            left_depth_pred, scale=1.0)
