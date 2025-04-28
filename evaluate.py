import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_loss = 0  # Initialize total loss

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(image)

            # Calculate loss
            if net.n_classes == 1:
                loss = F.binary_cross_entropy_with_logits(mask_pred.squeeze(1), mask_true.float())
                loss += dice_loss(torch.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                one_hot_masks = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                loss = F.cross_entropy(mask_pred, mask_true)
                loss += dice_loss(torch.softmax(mask_pred, dim=1).float(), one_hot_masks, multiclass=True)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], one_hot_masks[:, 1:], reduce_batch_first=False)

            total_loss += loss.item()  # Accumulate loss

    net.train()
    avg_dice_score = dice_score / max(num_val_batches, 1)
    avg_loss = total_loss / max(num_val_batches, 1)

    return avg_dice_score, avg_loss
