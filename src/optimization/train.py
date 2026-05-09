import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.dirname(os.path.dirname(current_dir))

from dataset import TEFilmDataset
from models import ThermoNetFusion

def train_model(args):
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Paths
    metadata_csv = os.path.join(project_root, 'data', 'simulations', 'metadata.csv')
    root_dir = os.path.join(project_root, 'data', 'simulations')
    
    if not os.path.exists(metadata_csv):
        print(f"Error: Database not found at {metadata_csv}. Please generate data first.")
        return

    print("Loading dataset (this may take a moment to scan the CSV)...")
    dataset = TEFilmDataset(metadata_csv=metadata_csv, root_dir=root_dir)
    total_samples = len(dataset)
    print(f"Total successful simulations found: {total_samples}")

    if total_samples == 0:
        print("Dataset is empty. Run generate_database.py first.")
        return

    # Train/Val/Test Split (80% / 10% / 10%)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model, Loss, Optimizer
    model = ThermoNetFusion(scalar_dim=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\nStarting Training Loop...")
    best_val_loss = float('inf')
    
    save_dir = os.path.join(project_root, 'results', 'models')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch_idx, (masks, scalars, targets) in enumerate(train_loader):
            masks, scalars, targets = masks.to(device), scalars.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(masks, scalars)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * masks.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx*len(masks)}/{len(train_dataset)}] Loss: {loss.item():.4f}")

        train_loss /= len(train_dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for masks, scalars, targets in val_loader:
                masks, scalars, targets = masks.to(device), scalars.to(device), targets.to(device)
                outputs = model(masks, scalars)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * masks.size(0)
                
        val_loss /= len(val_dataset)
        scheduler.step()
        
        print(f"==> Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'best_thermonet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"    [*] Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the 3D CNN Surrogate Model")
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of DataLoader workers')
    
    args = parser.parse_args()
    train_model(args)
