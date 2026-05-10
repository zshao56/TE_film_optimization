import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
project_root = os.path.dirname(os.path.dirname(current_dir))

from dataset import TEFilmDataset
from models import ThermoNetFusion

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def export_metadata_reports(dataset, split_subsets, report_dir):
    os.makedirs(report_dir, exist_ok=True)

    metadata = dataset.data_frame.copy()
    metadata.insert(0, 'dataset_index', np.arange(len(metadata)))
    metadata.insert(1, 'split', 'unused')

    for split_name, subset in split_subsets.items():
        metadata.loc[list(subset.indices), 'split'] = split_name

    metadata_path = os.path.join(report_dir, 'metadata_with_split.csv')
    metadata.to_csv(metadata_path, index=False)

    summary_cols = dataset.scalar_cols + [dataset.target_col]
    summary = metadata.groupby('split')[summary_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary_path = os.path.join(report_dir, 'metadata_split_summary.csv')
    summary.to_csv(summary_path, index=False)

    target_cols = [
        'split',
        f'{dataset.target_col}_count',
        f'{dataset.target_col}_mean',
        f'{dataset.target_col}_std',
        f'{dataset.target_col}_min',
        f'{dataset.target_col}_max'
    ]
    print("\nMetadata split target summary:")
    print(summary[target_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[INFO] Metadata with split labels saved to {metadata_path}")
    print(f"[INFO] Metadata split summary saved to {summary_path}")

def train_model(args):
    set_seed(args.seed)

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
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    report_dir = os.path.join(project_root, args.metadata_report_dir)
    export_metadata_reports(
        dataset,
        {'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
        report_dir
    )

    # DataLoaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': args.workers > 0
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Model, Loss, Optimizer
    model = ThermoNetFusion(scalar_dim=5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr
    )

    # TensorBoard Writer
    run_dir = os.path.join(project_root, 'runs', args.run_name)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"\n[INFO] TensorBoard logging enabled. Run `tensorboard --logdir runs` to view real-time charts.")

    print("\nStarting Training Loop...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
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
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            train_loss += loss.item() * masks.size(0)
            
            # Log batch loss to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx*len(masks)}/{len(train_dataset)}] Loss: {loss.item():.4f}")

        train_loss /= len(train_dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (masks, scalars, targets) in enumerate(val_loader):
                masks, scalars, targets = masks.to(device), scalars.to(device), targets.to(device)
                outputs = model(masks, scalars)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * masks.size(0)
                
                # Log validation batch loss to TensorBoard
                val_global_step = epoch * len(val_loader) + batch_idx
                writer.add_scalar('Loss/Val_Batch', loss.item(), val_global_step)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} [Val][{batch_idx*len(masks)}/{len(val_dataset)}] Loss: {loss.item():.4f}")
                
        val_loss /= len(val_dataset)
        previous_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch summaries to TensorBoard
        writer.add_scalar('Loss/Train_Epoch_Avg', train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch_Avg', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        lr_note = f" | LR: {current_lr:.6f}"
        if current_lr < previous_lr:
            lr_note += " (reduced)"
        print(f"==> Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{lr_note}")
        
        # Save Best Model
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_path = os.path.join(save_dir, 'best_thermonet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"    [*] Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"    [!] No validation improvement for {epochs_without_improvement}/{args.patience} epochs")
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered. Best Val Loss: {best_val_loss:.4f}")
                break

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the 3D CNN Surrogate Model")
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='AdamW weight decay')
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience in epochs')
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Minimum validation loss improvement')
    parser.add_argument('--lr-patience', type=int, default=5, help='Epochs without validation improvement before reducing LR')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='Learning rate reduction factor')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Lower bound for learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Max gradient norm; set <= 0 to disable')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    parser.add_argument('--run-name', type=str, default='thermonet_training', help='TensorBoard run directory under runs/')
    parser.add_argument('--metadata-report-dir', type=str, default=os.path.join('results', 'metadata'), help='Directory for split-labeled metadata reports')
    parser.add_argument('--workers', type=int, default=4, help='Number of DataLoader workers')
    
    args = parser.parse_args()
    train_model(args)
