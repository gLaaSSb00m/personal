import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Dataset
from torch import optim
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from vit_unet_torch import ViTUNet  # Import ViTUNet from vit_unet_torch.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Paths to the dataset
dir_img = Path('Pytorch_UNet\segmentation_full_body_mads_dataset_1192_img\images_preprocessed')
dir_mask = Path('Pytorch_UNet\segmentation_full_body_mads_dataset_1192_img\masks_preprocessed')
dir_checkpoint = Path('ceckpoints1/checkpoints_vit_unet/')
dir_plots = Path('plot1/plots_vit_unet/')

# Dataset class
class BasicDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.ids = [f.name for f in self.images_dir.glob('*.png')]  # Adjust extension if needed
        if not self.ids:
            raise ValueError(f"No images found in {images_dir}")
        # Check for paired masks
        for img_id in self.ids:
            if not (self.masks_dir / img_id).exists():
                raise ValueError(f"Mask not found for image {img_id}")
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_file = self.images_dir / self.ids[idx]
        mask_file = self.masks_dir / self.ids[idx]
        try:
            img = Image.open(img_file).convert('RGB')
            mask = Image.open(mask_file).convert('L')
        except Exception as e:
            raise ValueError(f"Error loading image/mask {img_file}/{mask_file}: {e}")

        if not np.isclose(self.scale, 1.0, atol=1e-9):
            img = img.resize((int(img.width * self.scale), int(img.height * self.scale)), Image.BILINEAR)
            mask = mask.resize((int(mask.width * self.scale), int(mask.height * self.scale)), Image.NEAREST)

        img = self.transform(img)
        mask = self.transform(np.array(mask))
        mask = (mask > 0).float()  # Ensure binary mask (0 or 1)
        return {'image': img, 'mask': mask}

# Dice coefficient and loss
def dice_coeff(pred, target):
    smooth = 1e-6
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def dice_loss(pred, target):
    return 1 - dice_coeff(pred, target)

# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    dice_score = 0
    accuracy = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.float32).squeeze(1)
            try:
                masks_pred = model(images).squeeze(1)
            except Exception as e:
                logging.error(f"Evaluation forward pass failed: {e}")
                raise
            masks_pred = torch.sigmoid(masks_pred)
            
            dice_score += dice_coeff(masks_pred, true_masks).item() * images.shape[0]
            pred_binary = (masks_pred > 0.5).float()
            accuracy += (pred_binary == true_masks).float().mean().item() * images.shape[0]
            total_samples += images.shape[0]
    
    model.train()
    return dice_score / total_samples, accuracy / total_samples

# Training function
def train_model(
    model,
    device,
    epochs=50,
    batch_size=4,
    learning_rate=1e-4,
    val_percent=0.1,
    img_scale=1.0,
    verbose=False,
):
    # 1. Load the dataset
    try:
        dataset = BasicDataset(dir_img, dir_mask, scale=img_scale)
        logging.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    # 2. Split into training and validation sets
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up optimizer, loss function, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 5. Lists to store metrics
    train_losses = []
    val_losses = []
    val_dice_scores = []
    val_accuracies = []

    # 6. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32).squeeze(1)

                # Forward pass
                try:
                    masks_pred = model(images, verbose=verbose).squeeze(1)
                except Exception as e:
                    logging.error(f"Training forward pass failed: {e}")
                    raise
                loss = criterion(masks_pred, true_masks)
                loss += dice_loss(torch.sigmoid(masks_pred), true_masks)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                # Consolidate postfix update
                unique_vals = torch.unique(true_masks, dim=None).cpu().numpy()
                pbar.set_postfix(loss=loss.item(), unique_mask_vals=unique_vals)

        # Compute average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        try:
            val_dice, val_accuracy = evaluate(model, val_loader, device)
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.float32).squeeze(1)
                masks_pred = model(images, verbose=False).squeeze(1)
                val_loss += criterion(masks_pred, true_masks).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(val_dice)
        val_accuracies.append(val_accuracy)

        # Update scheduler
        scheduler.step(val_dice)

        # Log metrics
        logging.info(f'Epoch {epoch + 1}/{epochs}: Training Loss={avg_train_loss:.4f}, '
                     f'Validation Loss={avg_val_loss:.4f}, Dice={val_dice:.4f}, Accuracy={val_accuracy:.4f}')

        # Save checkpoint
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        checkpoint_path = dir_checkpoint / f'vit_unet_checkpoint_epoch{epoch + 1}.pth'
        torch.save(model.state_dict(), str(checkpoint_path))
        logging.info(f'Checkpoint saved at {checkpoint_path}')

    # Plot and save curves
    dir_plots.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid()
    plt.savefig(dir_plots / 'loss_curve.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), val_dice_scores, label='Validation Dice Score')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Dice Score and Accuracy Curves')
    plt.legend()
    plt.grid()
    plt.savefig(dir_plots / 'metrics_curve.png')
    plt.close()

    return train_losses, val_losses, val_dice_scores, val_accuracies

# Main function
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Define the model
    try:
        model = ViTUNet().to(device)
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        raise

    # Test forward pass
    logging.info("Testing model forward pass...")
    try:
        test_input = torch.randn(1, 3, 256, 256).to(device)
        test_output = model(test_input, verbose=True)
        logging.info(f"Test output shape: {test_output.shape}")
    except Exception as e:
        logging.error(f"Model forward pass test failed: {e}")
        raise

    # Train the model
    try:
        train_losses, val_losses, val_dice_scores, val_accuracies = train_model(
            model=model,
            device=device,
            epochs=50,
            batch_size=4,
            learning_rate=1e-4,
            val_percent=0.1,
            img_scale=0.5,
            verbose=False  # Disable ViTUNet shape logging during training
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"Training Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")
    print(f"Validation Dice Score: {val_dice_scores[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")