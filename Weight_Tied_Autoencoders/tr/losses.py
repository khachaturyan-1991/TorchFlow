import torch
import torch.nn.functional as F


def dice_loss(y_true, y_pred, epsilon=1e-6):
    """Dice loss"""
    y_true = y_true.float()
    y_pred = y_pred.float()

    y_true = y_true.view(y_true.size(0), -1)
    y_pred = y_pred.view(y_pred.size(0), -1)

    intersection = torch.sum(y_true * y_pred, dim=1)
    union = torch.sum(y_true, dim=1) + torch.sum(y_pred, dim=1)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice

    return torch.mean(dice_loss)
