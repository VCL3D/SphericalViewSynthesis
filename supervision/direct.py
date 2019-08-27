import torch

def calculate_berhu_loss(pred, gt, mask, weights):
    diff = gt - pred
    abs_diff = torch.abs(diff)
    c = torch.max(abs_diff).item() / 5
    leq = (abs_diff <= c).float()
    l2_losses = (diff**2 + c**2) / (2 * c)
    loss = leq * abs_diff + (1 - leq) * l2_losses
    _, c, __, ___ = loss.size()
    count = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
    masked_loss = loss * mask.float()
    weighted_loss = masked_loss * weights
    return torch.mean(torch.sum(weighted_loss, dim=[1, 2, 3], keepdim=True) / count)