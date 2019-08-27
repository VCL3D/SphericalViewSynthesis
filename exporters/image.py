import torch
import cv2
import numpy

def save_image(filename, tensor, scale=255.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_depth(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.uint16(array)
        cv2.imwrite(filename.replace("#", str(n)), array)

def save_data(filename, tensor, scale=1000.0):
    b, _, __, ___ = tensor.size()
    for n in range(b):
        array = tensor[n, :, :, :].detach().cpu().numpy()
        array = array.transpose(1, 2, 0) * scale
        array = numpy.float32(array)
        cv2.imwrite(filename.replace("#", str(n)), array)
