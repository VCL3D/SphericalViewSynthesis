'''
    PyTorch implementation of https://github.com/google/layered-scene-inference
    accompanying the paper "Layer-structured 3D Scene Inference via View Synthesis", 
    ECCV 2018 https://shubhtuls.github.io/lsi/
'''

import torch

def __splat__(values, coords, splatted):
    b, c, h, w = splatted.size()
    uvs = coords
    u = uvs[:, 0, :, :].unsqueeze(1)
    v = uvs[:, 1, :, :].unsqueeze(1)
    
    u0 = torch.floor(u)
    u1 = u0 + 1
    v0 = torch.floor(v)
    v1 = v0 + 1

    u0_safe = torch.clamp(u0, 0.0, w-1)
    v0_safe = torch.clamp(v0, 0.0, h-1)
    u1_safe = torch.clamp(u1, 0.0, w-1)
    v1_safe = torch.clamp(v1, 0.0, h-1)

    u0_w = (u1 - u) * (u0 == u0_safe).detach().type(values.dtype)
    u1_w = (u - u0) * (u1 == u1_safe).detach().type(values.dtype)
    v0_w = (v1 - v) * (v0 == v0_safe).detach().type(values.dtype)
    v1_w = (v - v0) * (v1 == v1_safe).detach().type(values.dtype)

    top_left_w = u0_w * v0_w
    top_right_w = u1_w * v0_w
    bottom_left_w = u0_w * v1_w
    bottom_right_w = u1_w * v1_w

    weight_threshold = 1e-3
    top_left_w *= (top_left_w >= weight_threshold).detach().type(values.dtype)
    top_right_w *= (top_right_w >= weight_threshold).detach().type(values.dtype)
    bottom_left_w *= (bottom_left_w >= weight_threshold).detach().type(values.dtype)
    bottom_right_w *= (bottom_right_w >= weight_threshold).detach().type(values.dtype)

    for channel in range(c):
        top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
        top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
        bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
        bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w

        top_left_values = top_left_values.reshape(b, -1)
        top_right_values = top_right_values.reshape(b, -1)
        bottom_left_values = bottom_left_values.reshape(b, -1)
        bottom_right_values = bottom_right_values.reshape(b, -1)

        top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
        bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
        
        splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
        splatted_channel = splatted_channel.reshape(b, -1)
        splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
        splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
        splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
        splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)
    splatted = splatted.reshape(b, c, h, w)

def __weighted_average_splat__(depth, weights, epsilon=1e-8):
    zero_weights = (weights <= epsilon).detach().type(depth.dtype)
    return depth / (weights + epsilon * zero_weights)

def __depth_distance_weights__(depth, max_depth=20.0):
    weights = 1.0 / torch.exp(2 * depth / max_depth)
    return weights

def render(img, depth, coords, max_depth=20.0):
    splatted_img = torch.zeros_like(img)
    splatted_wgts = torch.zeros_like(depth)        
    weights = __depth_distance_weights__(depth, max_depth=max_depth)
    __splat__(img * weights, coords, splatted_img)
    __splat__(weights, coords, splatted_wgts)
    recon = __weighted_average_splat__(splatted_img, splatted_wgts)
    mask = (splatted_wgts > 1e-3).detach()
    return recon, mask

def render_to(src, tgt, wgts, depth, coords, max_depth=20.0):    
    weights = __depth_distance_weights__(depth, max_depth=max_depth)
    __splat__(src * weights, coords, tgt)
    __splat__(weights, coords, wgts)
    tgt = __weighted_average_splat__(tgt, wgts)
    mask = (wgts > 1e-3).detach()
    return mask