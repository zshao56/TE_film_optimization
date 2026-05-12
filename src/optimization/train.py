import os
import sys
import argparse
import random
import numpy as np
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


def unpack_batch(batch):
    if len(batch) == 4:
        masks, scalars, targets, weights = batch
        return masks, scalars, targets, weights
    masks, scalars, targets = batch
    return masks, scalars, targets, None


def weighted_mse_loss(outputs, targets, weights=None):
    loss = (outputs - targets) ** 2
    if weights is None:
        return loss.mean()
    return (loss * weights).sum() / weights.sum().clamp_min(1e-12)


def resolve_target_cutoff(dataset, quantile):
    if not 0.0 < quantile < 1.0:
        raise ValueError("underpredict_quantile must be between 0 and 1.")
    raw_cutoff = float(dataset.data_frame[dataset.target_col].quantile(quantile))
    if dataset.normalize_target:
        loss_cutoff = (raw_cutoff - dataset.target_mean) / dataset.target_std
    else:
        loss_cutoff = raw_cutoff
    return raw_cutoff, float(loss_cutoff)


def regression_loss(
    outputs,
    targets,
    weights=None,
    underpredict_penalty=0.0,
    underpredict_cutoff=None,
):
    base_loss = weighted_mse_loss(outputs, targets, weights)
    under_loss = outputs.new_tensor(0.0)

    if underpredict_penalty > 0.0:
        if underpredict_cutoff is None:
            raise ValueError("underpredict_cutoff is required when underpredict_penalty > 0.")
        top_mask = targets >= underpredict_cutoff
        if top_mask.any().item():
            under_error = torch.relu(targets - outputs)
            under_loss = (under_error[top_mask] ** 2).mean()

    total_loss = base_loss + underpredict_penalty * under_loss
    return total_loss, base_loss, under_loss


def train_model(args):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard is required for training logs. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    set_seed(args.seed)
    if args.underpredict_penalty < 0.0:
        raise ValueError("underpredict_penalty must be >= 0.")

    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Paths
    metadata_csv = args.metadata_csv or os.path.join(project_root, 'data', 'simulations', 'metadata.csv')
    root_dir = args.root_dir or os.path.join(project_root, 'data', 'simulations')
    
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Database not found at {metadata_csv}. Please generate and filter data first.")

    print("Loading dataset (this may take a moment to scan the CSV)...")
    dataset = TEFilmDataset(
        metadata_csv=metadata_csv,
        root_dir=root_dir,
        normalize_target=args.normalize_target,
        return_weight=args.top_weight > 1.0,
        top_quantile=args.top_quantile,
        top_weight=args.top_weight,
        include_boundary_channel=args.include_boundary_channel,
    )
    total_samples = len(dataset)
    print(f"Total successful simulations found: {total_samples}")

    if total_samples == 0:
        raise RuntimeError("Dataset is empty. Run generate_database.py and metadata filtering first.")

    # Train/Val/Test Split (80% / 10% / 10%)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    if args.normalize_target:
        print(f"Target normalization enabled: mean={dataset.target_mean:.6g}, std={dataset.target_std:.6g}")
    if dataset.top_weight_cutoff is not None:
        print(
            f"Top-region weighted loss enabled: delta_T >= {dataset.top_weight_cutoff:.6g} K "
            f"gets weight {args.top_weight:.3g}"
        )
    underpredict_raw_cutoff = None
    underpredict_loss_cutoff = None
    underpredict_quantile = None
    if args.underpredict_penalty > 0.0:
        underpredict_quantile = (
            args.underpredict_quantile
            if args.underpredict_quantile is not None
            else args.top_quantile
        )
        underpredict_raw_cutoff, underpredict_loss_cutoff = resolve_target_cutoff(
            dataset,
            underpredict_quantile,
        )
        print(
            f"High-delta-T underprediction penalty enabled: true delta_T >= "
            f"{underpredict_raw_cutoff:.6g} K gets penalty {args.underpredict_penalty:.3g}"
        )

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
    print(f"Scalar inputs ({len(dataset.scalar_cols)}): {dataset.scalar_cols}")
    print(f"3D input channels: {dataset.input_channels}")
    model = ThermoNetFusion(scalar_dim=len(dataset.scalar_cols), input_channels=dataset.input_channels).to(device)
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
        train_base_loss = 0.0
        train_under_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            masks, scalars, targets, weights = unpack_batch(batch)
            masks, scalars, targets = masks.to(device), scalars.to(device), targets.to(device)
            if weights is not None:
                weights = weights.to(device)
            
            optimizer.zero_grad()
            outputs = model(masks, scalars)
            loss, base_loss, under_loss = regression_loss(
                outputs,
                targets,
                weights=weights,
                underpredict_penalty=args.underpredict_penalty,
                underpredict_cutoff=underpredict_loss_cutoff,
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            train_loss += loss.item() * masks.size(0)
            train_base_loss += base_loss.item() * masks.size(0)
            train_under_loss += under_loss.item() * masks.size(0)
            
            # Log batch loss to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            writer.add_scalar('Loss/Train_Base_MSE_Batch', base_loss.item(), global_step)
            if args.underpredict_penalty > 0.0:
                writer.add_scalar('Loss/Train_Underpredict_Batch', under_loss.item(), global_step)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} [{batch_idx*len(masks)}/{len(train_dataset)}] Loss: {loss.item():.4f}")

        train_loss /= len(train_dataset)
        train_base_loss /= len(train_dataset)
        train_under_loss /= len(train_dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_base_loss = 0.0
        val_under_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                masks, scalars, targets, _weights = unpack_batch(batch)
                masks, scalars, targets = masks.to(device), scalars.to(device), targets.to(device)
                outputs = model(masks, scalars)
                loss, base_loss, under_loss = regression_loss(
                    outputs,
                    targets,
                    underpredict_penalty=args.underpredict_penalty,
                    underpredict_cutoff=underpredict_loss_cutoff,
                )
                val_loss += loss.item() * masks.size(0)
                val_base_loss += base_loss.item() * masks.size(0)
                val_under_loss += under_loss.item() * masks.size(0)
                
                # Log validation batch loss to TensorBoard
                val_global_step = epoch * len(val_loader) + batch_idx
                writer.add_scalar('Loss/Val_Batch', loss.item(), val_global_step)
                writer.add_scalar('Loss/Val_Base_MSE_Batch', base_loss.item(), val_global_step)
                if args.underpredict_penalty > 0.0:
                    writer.add_scalar('Loss/Val_Underpredict_Batch', under_loss.item(), val_global_step)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs} [Val][{batch_idx*len(masks)}/{len(val_dataset)}] Loss: {loss.item():.4f}")
                
        val_loss /= len(val_dataset)
        val_base_loss /= len(val_dataset)
        val_under_loss /= len(val_dataset)
        previous_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch summaries to TensorBoard
        writer.add_scalar('Loss/Train_Epoch_Avg', train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch_Avg', val_loss, epoch)
        writer.add_scalar('Loss/Train_Base_MSE_Epoch_Avg', train_base_loss, epoch)
        writer.add_scalar('Loss/Val_Base_MSE_Epoch_Avg', val_base_loss, epoch)
        if args.underpredict_penalty > 0.0:
            writer.add_scalar('Loss/Train_Underpredict_Epoch_Avg', train_under_loss, epoch)
            writer.add_scalar('Loss/Val_Underpredict_Epoch_Avg', val_under_loss, epoch)
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
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'normalize_target': args.normalize_target,
                    'target_mean': float(dataset.target_mean),
                    'target_std': float(dataset.target_std),
                    'scalar_cols': dataset.scalar_cols,
                    'scalar_dim': len(dataset.scalar_cols),
                    'input_channels': dataset.input_channels,
                    'include_boundary_channel': args.include_boundary_channel,
                    'target_col': dataset.target_col,
                    'seed': args.seed,
                    'top_quantile': args.top_quantile,
                    'top_weight': args.top_weight,
                    'top_weight_cutoff': dataset.top_weight_cutoff,
                    'underpredict_penalty': args.underpredict_penalty,
                    'underpredict_quantile': underpredict_quantile,
                    'underpredict_cutoff': underpredict_raw_cutoff,
                },
                save_path,
            )
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
    parser.add_argument('--metadata-csv', type=str, default=None, help='Path to metadata.csv')
    parser.add_argument('--root-dir', type=str, default=None, help='Directory containing the fields/ folder')
    parser.add_argument('--workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--normalize-target', action='store_true', help='Train on Z-score normalized delta_T targets')
    parser.add_argument('--top-quantile', type=float, default=0.9, help='High delta_T quantile used for optional weighted loss')
    parser.add_argument('--top-weight', type=float, default=1.0, help='Loss weight for samples above --top-quantile; 1 disables weighting')
    parser.add_argument('--underpredict-penalty', type=float, default=0.0, help='Extra loss coefficient for underpredicting high-delta-T samples; 0 disables it')
    parser.add_argument('--underpredict-quantile', type=float, default=None, help='Quantile cutoff for underprediction penalty; defaults to --top-quantile')
    parser.add_argument('--include-boundary-channel', action='store_true', help='Add the hot-boundary temperature map as a second 3D CNN input channel')
    
    args = parser.parse_args()
    train_model(args)
