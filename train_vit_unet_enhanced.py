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
dir_img = Path('segmentation_full_body_mads_dataset_1192_img\images')
dir_mask = Path('segmentation_full_body_mads_dataset_1192_img\masks')
dir_checkpoint = Path('./checkpoints_vit_unet/')
dir_plots = Path('./plots_vit_unet2/')

# Dataset class with separate transforms for images and masks
class BasicDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = 1.0, img_transform=None, mask_transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.ids = [f.name for f in self.images_dir.glob('*.png')]  # Adjust extension if needed
        if not self.ids:
            raise ValueError(f"No images found in {images_dir}")
        # Check for paired masks
        for img_id in self.ids:
            if not (self.masks_dir / img_id).exists():
                raise ValueError(f"Mask not found for image {img_id}")

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

        # Apply transforms if provided
        if self.img_transform and self.mask_transform:
            # Ensure consistent random transformations for image and mask where applicable
            seed = np.random.randint(2147483647)  # Large random seed
            torch.manual_seed(seed)
            img = self.img_transform(img)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            img = T.ToTensor()(img)
            mask = T.ToTensor()(np.array(mask))

        mask = (mask > 0).float()  # Ensure binary mask (0 or 1)
        return {'image': img, 'mask': mask}

# Function to compute class imbalance for pos_weight
def compute_pos_weight(dataset, device):
    total_pixels = 0
    positive_pixels = 0
    for i in range(len(dataset)):
        mask = dataset[i]['mask']
        total_pixels += mask.numel()
        positive_pixels += mask.sum().item()
    negative_pixels = total_pixels - positive_pixels
    pos_weight = negative_pixels / positive_pixels if positive_pixels > 0 else 1.0
    return torch.tensor(pos_weight).to(device)

# Dice coefficient and loss
def dice_coeff(pred, target):
    smooth = 1e-6
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def dice_loss(pred, target):
    return 1 - dice_coeff(pred, target)

# Evaluation function (for both training and validation)
def evaluate(model, dataloader, device, criterion):
    model.eval()
    dice_score = 0
    accuracy = 0
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            images, true_masks = batch['image'], batch['mask']
            images = images.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.float32).squeeze(1)
            try:
                masks_pred = model(images).squeeze(1)  # Raw logits
            except Exception as e:
                logging.error(f"Evaluation forward pass failed: {e}")
                raise
            # Compute loss
            loss = criterion(masks_pred, true_masks)
            total_loss += loss.item() * images.shape[0]
            
            # Compute metrics
            masks_pred_sigmoid = torch.sigmoid(masks_pred)  # Apply sigmoid for metrics
            dice_score += dice_coeff(masks_pred_sigmoid, true_masks).item() * images.shape[0]
            pred_binary = (masks_pred_sigmoid > 0.5).float()
            accuracy += (pred_binary == true_masks).float().mean().item() * images.shape[0]
            total_samples += images.shape[0]
    
    model.train()
    avg_loss = total_loss / total_samples
    avg_dice = dice_score / total_samples
    avg_accuracy = accuracy / total_samples
    return avg_loss, avg_dice, avg_accuracy

# Learning rate warm-up scheduler
class WarmupLR:
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_epoch = 0

    def step(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.final_lr - self.base_lr) * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return self.base_lr + (self.final_lr - self.base_lr) * self.current_epoch / self.warmup_epochs
        return self.final_lr

# Training function
def train_model(
    model,
    device,
    epochs=10,
    batch_size=10,
    learning_rate=1e-4,
    val_percent=0.2,
    img_scale=0.5,
    verbose=False,
):
    # 1. Define transforms for training and validation
    # ImageNet normalization for ViT (only for images)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_img_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        T.RandomResizedCrop(size=(int(256 * img_scale), int(256 * img_scale)), scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        normalize
    ])
    train_mask_transform = T.Compose([
        T.RandomHorizontalFlip(),  # Same as image
        T.RandomRotation(20),      # Same as image
        T.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Same as image
        T.RandomResizedCrop(size=(int(256 * img_scale), int(256 * img_scale)), scale=(0.8, 1.0)),  # Same as image
        T.ToTensor()  # No normalization for masks
    ])
    val_img_transform = T.Compose([
        T.Resize((int(256 * img_scale), int(256 * img_scale))),
        T.ToTensor(),
        normalize
    ])
    val_mask_transform = T.Compose([
        T.Resize((int(256 * img_scale), int(256 * img_scale))),
        T.ToTensor()  # No normalization for masks
    ])

    # 2. Load the dataset with appropriate transforms
    try:
        train_dataset = BasicDataset(
            dir_img, dir_mask, scale=img_scale,
            img_transform=train_img_transform, mask_transform=train_mask_transform
        )
        val_dataset = BasicDataset(
            dir_img, dir_mask, scale=img_scale,
            img_transform=val_img_transform, mask_transform=val_mask_transform
        )
        logging.info(f"Loaded dataset with {len(train_dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    # 3. Split into training and validation sets
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set.dataset = train_dataset  # Ensure transform is applied
    val_set.dataset = val_dataset

    # 4. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 5. Compute pos_weight based on dataset
    pos_weight = compute_pos_weight(train_set, device)
    logging.info(f"Computed pos_weight: {pos_weight.item():.4f}")

    # 6. Set up optimizer, loss function, and schedulers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    warmup_scheduler = WarmupLR(optimizer, warmup_epochs=5, base_lr=1e-5, final_lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 7. Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_dice_scores = []
    val_accuracies = []

    # 8. Early stopping parameters
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    # 9. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        total_samples = 0
        warmup_scheduler.step(epoch)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32).squeeze(1)

                # Forward pass
                try:
                    masks_pred = model(images, verbose=verbose).squeeze(1)  # Raw logits
                except Exception as e:
                    logging.error(f"Training forward pass failed: {e}")
                    raise
                bce_loss = criterion(masks_pred, true_masks)
                dice = dice_loss(torch.sigmoid(masks_pred), true_masks)
                loss = 0.3 * bce_loss + 0.7 * dice  # Weighted combination

                # Compute training accuracy for this batch
                masks_pred_sigmoid = torch.sigmoid(masks_pred)
                pred_binary = (masks_pred_sigmoid > 0.5).float()
                batch_accuracy = (pred_binary == true_masks).float().mean().item()
                epoch_accuracy += batch_accuracy * images.shape[0]

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pbar.update(images.shape[0])
                epoch_loss += loss.item() * images.shape[0]
                total_samples += images.shape[0]
                unique_vals = torch.unique(true_masks, dim=None).cpu().numpy()
                pbar.set_postfix(loss=loss.item(), accuracy=batch_accuracy, unique_mask_vals=unique_vals)

        # Compute average training metrics
        avg_train_loss = epoch_loss / total_samples
        avg_train_accuracy = epoch_accuracy / total_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # Validation
        try:
            avg_val_loss, val_dice, avg_val_accuracy = evaluate(model, val_loader, device, criterion)
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            raise
        val_losses.append(avg_val_loss)
        val_dice_scores.append(val_dice)
        val_accuracies.append(avg_val_accuracy)

        # Update scheduler
        scheduler.step(val_dice)

        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch + 1}/{epochs}: '
                     f'Training Loss={avg_train_loss:.4f}, Training Accuracy={avg_train_accuracy:.4f}, '
                     f'Validation Loss={avg_val_loss:.4f}, Validation Accuracy={avg_val_accuracy:.4f}, '
                     f'Dice={val_dice:.4f}, Learning Rate={current_lr:.6f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            # Save the best model
            best_checkpoint_path = dir_checkpoint / 'vit_unet_best.pth'
            torch.save(model.state_dict(), str(best_checkpoint_path))
            logging.info(f'Best model saved at {best_checkpoint_path}')
        else:
            counter += 1
        if counter >= patience:
            logging.info(f'Early stopping at epoch {epoch + 1}')
            break

        # Save checkpoint for the current epoch
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        checkpoint_path = dir_checkpoint / f'vit_unet_checkpoint_epoch{epoch + 1}.pth'
        torch.save(model.state_dict(), str(checkpoint_path))
        logging.info(f'Checkpoint saved at {checkpoint_path}')

    # Plot and save curves
    dir_plots.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid()
    plt.savefig(dir_plots / 'loss_curve.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    plt.grid()
    plt.savefig(dir_plots / 'accuracy_curve.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_dice_scores) + 1), val_dice_scores, label='Validation Dice Score')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Validation Dice Score and Accuracy Curves')
    plt.legend()
    plt.grid()
    plt.savefig(dir_plots / 'metrics_curve.png')
    plt.close()

    return train_losses, val_losses, val_dice_scores, val_accuracies, train_accuracies

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
        train_losses, val_losses, val_dice_scores, val_accuracies, train_accuracies = train_model(
            model=model,
            device=device,
            epochs=10,
            batch_size=10,
            learning_rate=1e-4,
            val_percent=0.2,
            img_scale=0.5,
            verbose=False
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"Training Loss: {train_losses[-1]:.4f}")
    print(f"Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Validation Dice Score: {val_dice_scores[-1]:.4f}")