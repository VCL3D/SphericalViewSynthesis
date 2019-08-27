import torch
import numpy

def create_image_grid(width, height, data_type=torch.float32):        
    v_range = (
        torch.arange(0, height) # [0 - h]
        .view(1, height, 1) # [1, [0 - h], 1]
        .expand(1, height, width) # [1, [0 - h], W]
        .type(data_type)  # [1, H, W]
    )
    u_range = (
        torch.arange(0, width) # [0 - w]
        .view(1, 1, width) # [1, 1, [0 - w]]
        .expand(1, height, width) # [1, H, [0 - w]]
        .type(data_type)  # [1, H, W]
    )
    return torch.stack((u_range, v_range), dim=1)  # [1, 2, H, W]

def coord_u(uvgrid):
    return uvgrid[:, 0, :, :].unsqueeze(1)

def coord_v(uvgrid):
    return uvgrid[:, 1, :, :].unsqueeze(1)

def create_spherical_grid(width, horizontal_shift=(-numpy.pi - numpy.pi / 2.0),
    vertical_shift=(-numpy.pi / 2.0), data_type=torch.float32):
    height = int(width // 2.0)
    v_range = (
        torch.arange(0, height) # [0 - h]
        .view(1, height, 1) # [1, [0 - h], 1]
        .expand(1, height, width) # [1, [0 - h], W]
        .type(data_type)  # [1, H, W]
    )
    u_range = (
        torch.arange(0, width) # [0 - w]
        .view(1, 1, width) # [1, 1, [0 - w]]
        .expand(1, height, width) # [1, H, [0 - w]]
        .type(data_type)  # [1, H, W]
    )
    u_range *= (2 * numpy.pi / width) # [0, 2 * pi]
    v_range *= (numpy.pi / height) # [0, pi]
    u_range += horizontal_shift # [-hs, 2 * pi - hs] -> standard values are [-3 * pi / 2, pi / 2]
    v_range += vertical_shift # [-vs, pi - vs] -> standard values are [-pi / 2, pi / 2]
    return torch.stack((u_range, v_range), dim=1)  # [1, 2, H, W]

def phi(sgrid): # longitude or azimuth
    return sgrid[:, 0, :, :].unsqueeze(1)

def azimuth(sgrid): # longitude or phi
    return sgrid[:, 0, :, :].unsqueeze(1)

def longitude(sgrid): # phi or azimuth
    return sgrid[:, 0, :, :].unsqueeze(1)

def theta(sgrid): # latitude or elevation
    return sgrid[:, 1, :, :].unsqueeze(1)

def elevation(sgrid): # theta or elevation
    return sgrid[:, 1, :, :].unsqueeze(1)

def latitude(sgrid): # latitude or theta
    return sgrid[:, 1, :, :].unsqueeze(1)