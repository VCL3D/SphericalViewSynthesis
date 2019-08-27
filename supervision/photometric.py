import torch

from .ssim import *

class PhotometricLossParameters(object):
    def __init__(self, alpha=0.85, l1_estimator='none',\
        ssim_estimator='none', window=7, std=1.5, ssim_mode='gaussian'):
        super(PhotometricLossParameters, self).__init__()
        self.alpha = alpha
        self.l1_estimator = l1_estimator
        self.ssim_estimator = ssim_estimator
        self.window = window
        self.std = std
        self.ssim_mode = ssim_mode

    def get_alpha(self):
        return self.alpha

    def get_l1_estimator(self):
        return self.l1_estimator

    def get_ssim_estimator(self):
        return self.ssim_estimator

    def get_window(self):
        return self.window

    def get_std(self):
        return self.std

    def get_ssim_mode(self):
        return self.ssim_mode

def calculate_loss(pred, gt, params, mask, weights):
    valid_mask = mask.type(gt.dtype)
    masked_gt = gt * valid_mask
    masked_pred = pred * valid_mask
    l1 = torch.abs(masked_gt - masked_pred)
    d_ssim = torch.clamp(
        (
            1 - ssim_loss(masked_pred, masked_gt, kernel_size=params.get_window(),
                std=params.get_std(), mode=params.get_ssim_mode())
        ) / 2, 0, 1)
    loss = (
        d_ssim * params.get_alpha()
        + l1 * (1 - params.get_alpha())
    )    
    loss *= valid_mask
    loss *= weights        
    count = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    return torch.mean(torch.sum(loss, dim=[1, 2, 3], keepdim=True) / count)
