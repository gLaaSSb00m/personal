import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from unet.unet_model import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from torchsummary import summary
import matplotlib.pyplot as plt  # Add this import at the top of the file

# Paths to the dataset
dir_img = Path('segmentation_full_body_mads_dataset_1192_img\images')
dir_mask = Path('segmentation_full_body_mads_dataset_1192_img\masks')
dir_checkpoint = Path('./checkpoints/')

# Training function
def train_model(
    model,
    device,
    epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    val_percent=0.1,
    img_scale=1.0,
):
    # 1. Load the dataset
    dataset = BasicDataset(dir_img, dir_mask, scale=img_scale)

    # 2. Split into training and validation sets
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up optimizer, loss function, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    criterion = torch.nn.CrossEntropyLoss() if model.n_classes > 1 else torch.nn.BCEWithLogitsLoss()

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # 5. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        total_samples = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # Ensure true_masks has the correct shape
                true_masks = true_masks.squeeze(1)

                # Forward pass
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                # Add Dice loss for binary or multiclass segmentation
                if model.n_classes == 1:
                    loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                else:
                    one_hot_masks = torch.nn.functional.one_hot(true_masks, num_classes=model.n_classes)
                    one_hot_masks = one_hot_masks.permute(0, 3, 1, 2).float()
                    loss += dice_loss(torch.softmax(masks_pred, dim=1).float(), one_hot_masks, multiclass=True)

                # Compute accuracy
                with torch.no_grad():
                    if model.n_classes == 1:
                        pred_binary = (torch.sigmoid(masks_pred) > 0.5).float()
                        batch_accuracy = (pred_binary == true_masks.float()).float().mean().item()
                    else:
                        pred_classes = torch.argmax(masks_pred, dim=1)
                        batch_accuracy = (pred_classes == true_masks).float().mean().item()

                epoch_accuracy += batch_accuracy * images.shape[0]
                total_samples += images.shape[0]

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'accuracy (batch)': batch_accuracy})

        # Compute average training metrics for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_accuracy = epoch_accuracy / total_samples
        train_losses.append(avg_train_loss)

        # Validation
        val_score, val_loss = evaluate(model, val_loader, device, amp=False)
        val_losses.append(val_loss)  # Append the validation loss for plotting
        scheduler.step(val_score)  # Pass only the validation Dice score to the scheduler

        logging.info(f'Epoch {epoch + 1}: '
                     f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, '
                     f'Val Loss: {val_loss:.4f}, Val Dice: {val_score:.4f}')

        # Save checkpoint
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch + 1}.pth'))
        logging.info(f'Checkpoint {epoch + 1} saved!')

    return train_losses, val_losses

# Main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    n_channels = 3  # RGB images
    n_classes = 2   # Binary segmentation
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    model.to(device=device)

    # Display model summary
    summary(model, input_size=(n_channels, 256, 256))

    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        device=device,
        epochs=10,
        batch_size=4,
        learning_rate=1e-4,
        val_percent=0.1,
        img_scale=0.5,
    )

    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()