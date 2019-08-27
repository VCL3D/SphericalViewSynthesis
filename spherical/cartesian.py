import torch

from .grid import *

'''
    Cartesian coordinates extraction from Spherical coordinates
        z is forward axis
        y is the up axis
        x is the right axis
        r is the radius (i.e. spherical depth)
        phi is the longitude/azimuthial rotation angle (defined on the x-z plane)
        theta is the latitude/elevation rotation angle (defined on the y-z plane)
'''
def coord_x(sgrid, depth):
    return ( # r * sin(phi) * sin(theta) -> r * cos(phi) * -cos(theta) in our offsets
        depth # this is due to the offsets as explained below
        * torch.cos(phi(sgrid)) # long = x - 3 * pi / 2
        * -1 * torch.cos(theta(sgrid)) # lat = y - pi / 2
    )

def coord_y(sgrid, depth):
    return ( # r * cos(theta) -> r * sin(theta) in our offsets
        depth # this is due to the offsets as explained below
        * torch.sin(theta(sgrid)) # lat = y - pi / 2
    )

def coord_z(sgrid, depth):
    return ( # r * cos(phi) * sin(theta) -> r * -sin(phi) * -cos(theta) in our offsets
        depth # this is due to the offsets as explained above
        * torch.sin(phi(sgrid)) # * -1
        * torch.cos(theta(sgrid)) # * -1
    ) # the -1s cancel out

def coords_3d(sgrid, depth):
    return torch.cat(
        (
            coord_x(sgrid, depth),
            coord_y(sgrid, depth),
            coord_z(sgrid, depth)
        ), dim=1
    )

def xi(pcloud):
    return pcloud[:, 0, :, :].unsqueeze(1)

def yi(pcloud):
    return pcloud[:, 1, :, :].unsqueeze(1)

def zeta(pcloud):
    return pcloud[:, 2, :, :].unsqueeze(1)