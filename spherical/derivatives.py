import torch

from .grid import *
from .cartesian import *

''' Image (I) spatial derivatives '''
def dI_du(img):
    right_pad = (0, 1, 0, 0)
    tensor = torch.nn.functional.pad(img, right_pad, mode="replicate")
    gu = tensor[:, :, :, :-1] - tensor[:, :, :, 1:]  # NCHW
    return gu

def dI_dv(img):
    bottom_pad = (0, 0, 0, 1)
    tensor = torch.nn.functional.pad(img, bottom_pad, mode="replicate")
    dv = tensor[:, :, :-1, :] - tensor[:, :, 1:, :]  # NCHW
    return dv

def dI_duv(img):
    du = dI_du(img)
    dv = dI_dv(img)
    duv = torch.cat((du, dv), dim=1)
    duv_mag = torch.norm(duv, p=2, dim=1, keepdim=True)
    return duv_mag

''' 
    Spherical coordinates (r, phi, theta) derivatives 
    w.r.t. their Cartesian counterparts (x, y, z)
'''
def dr_dx(sgrid):
    return ( # sin(lat) * sin(long) -> cos(long) * -cos(lat)
        -1 # this is due to the offsets as explaned below
        * torch.cos(phi(sgrid)) # long = x - 3 * pi / 2
        * torch.cos(theta(sgrid)) # lat = y - pi / 2
    ) # the depth (radius) distortion for each spherical coord with a horizontal baseline

def dphi_dx(sgrid):
    return ( # cos(long) / sin(lat) -> -sin(long) / -cos(lat)
        torch.sin(phi(sgrid)) # * -1
        / torch.cos(theta(sgrid)) # * -1
    ) # the -1s cancel out and are ommitted

def dtheta_dx(sgrid):
    return ( # sin(long) * cos(lat) -> cos(long) * sin(lat)
        torch.cos(phi(sgrid)) * torch.sin(theta(sgrid))
    )

def dtheta_dy(sgrid):
    return ( # -sin(lat) -> -1 * -cos(lat) == cos(lat)
        torch.cos(theta(sgrid))
    )

def dphi_horizontal(sgrid, depth, baseline):
    _, __, h, ___ = depth.size()
    return torch.clamp(    
        (
            torch.sin(phi(sgrid))
            / (
                depth
                * torch.cos(theta(sgrid))    
            )
            * baseline
            * (h / numpy.pi)
        ),
        -h, h # h = w/2 the max disparity due to our spherical nature (i.e. front/back symmetry)
    )

def dtheta_horizontal(sgrid, depth, baseline):
    _, __, h, ___ = depth.size()
    return torch.clamp(
        (
            torch.cos(phi(sgrid))
            * torch.sin(theta(sgrid))
            * baseline
            / depth
            * (h / numpy.pi)
        ), 
        0, h
    )

def dr_horizontal(sgrid, baseline):
    return ( # sin(lat) * sin(long) -> cos(long) * -cos(lat)
        -1 # this is due to the offsets as explained below
        * torch.cos(phi(sgrid)) # long = x - 3 * pi / 2
        * torch.cos(theta(sgrid)) # lat = y - pi / 2
        * baseline
    ) # the depth (radius) distortion for each spherical coord with a horizontal baseline

def dtheta_vertical(sgrid, depth, baseline):
    _, __, h, ___ = depth.size()
    return (
        torch.cos(theta(sgrid))
        * baseline
        / depth
        * (h / numpy.pi)
    )

'''
    Structured Point Cloud Vertices (V) spatial derivatives
'''
def dV_dx(pcloud):
    return dI_duv(xi(pcloud))

def dV_dy(pcloud):
    return dI_duv(yi(pcloud))

def dV_dz(pcloud):
    return dI_duv(zeta(pcloud))

def dV_dxyz(pcloud):
    du_x = dI_du(xi(pcloud))
    dv_x = dI_dv(xi(pcloud))
    
    du_y = dI_du(yi(pcloud))
    dv_y = dI_dv(yi(pcloud))
    
    du_z = dI_du(zeta(pcloud))
    dv_z = dI_dv(zeta(pcloud))
    
    du_xyz = torch.abs(du_x) + torch.abs(du_y) + torch.abs(du_z) 
    dv_xyz = torch.abs(dv_x) + torch.abs(dv_y) + torch.abs(dv_z)

    duv_xyz = torch.cat((du_xyz, dv_xyz), dim=1)
    duv__xyz_mag = torch.norm(duv_xyz, p=2, dim=1, keepdim=True)
    return duv__xyz_mag