import os
import sys
import time
import argparse
import logging
import json
import torch
from torchsummary import summary
import numpy as np
from datetime import datetime
from pathlib import Path

# Add module directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.data.genomic_sequences import GenomicSequenceDataset
from src.models.cnn_standard import DNACNN as StandardCNN
from src.models.cnn_advanced import DNACNN as AdvancedCNN
from src.models.naive_bayes import MultinomialNaiveBayes, train_nb, evaluate_nb
from src.models.naive_bayes import load_data as load_nb_data
from src.models.cnn_advanced import train as train_advanced
from src.models.cnn_advanced import evaluate as evaluate_advanced
from src.models.cnn_standard import train as train_standard
from src.models.cnn_standard import evaluate as evaluate_standard


def setup_logging(log_dir="logs", model_name="model"):
    """Set up logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(state, filename, is_best=False):
    """Save checkpoint to disk."""
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(".pt", "_best.pt")
        torch.save(state, best_filename)
        logging.info(f"Saved best model to {best_filename}")


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load checkpoint from disk."""
    if not os.path.exists(filename):
        return None, 0, 0, None
    
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    history = checkpoint.get('history', None)
    
    return model, start_epoch, best_val_loss, history


def prepare_dataloader(args):
    """Create train, validation, and test dataloaders."""
    if args.model_type == "naive_bayes":
        # Naive Bayes uses different data loading
        return load_nb_data(args.data_dir)
    
    # For CNN models
    train_dataset = GenomicSequenceDataset(
        split_dir=args.data_dir,
        split_type="train",
        window_size=args.window_size,
        stride=args.stride,
        cache_size=args.cache_size
    )
    
    val_dataset = GenomicSequenceDataset(
        split_dir=args.data_dir,
        split_type="val",
        window_size=args.window_size,
        stride=args.stride,
        cache_size=args.cache_size
    )
    
    test_dataset = GenomicSequenceDataset(
        split_dir=args.data_dir,
        split_type="test",
        window_size=args.window_size,
        stride=args.stride,
        cache_size=args.cache_size
    )
    
    if args.sample_limit > 0:
        train_dataset.sample_regions(n_regions=args.sample_limit)
        val_dataset.sample_regions(n_regions=args.sample_limit // 10)
        test_dataset.sample_regions(n_regions=args.sample_limit // 10)
    
    # Create dataloaders
    train_loader = train_dataset.get_dataloader(
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    test_loader = test_dataset.get_dataloader(
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Calculate number of classes
    n_classes = len(train_dataset.encoder.classes_)
    logging.info(f"Number of classes: {n_classes}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'n_classes': n_classes,
        'train_dataset': train_dataset
    }


def create_model(args, n_classes):
    """Create the model based on model_type argument."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.model_type == "naive_bayes":
        return MultinomialNaiveBayes(), device
    
    elif args.model_type == "cnn_standard":
        model = StandardCNN(
            n_classes=n_classes,
            seq_length=args.window_size,
            n_channels=5,
            dropout=args.dropout
        )
    
    elif args.model_type == "cnn_advanced":
        model = AdvancedCNN(
            n_classes=n_classes,
            seq_length=args.window_size,
            n_channels=5,
            conv_channels=[args.conv1_channels, args.conv2_channels, args.conv3_channels],
            kernel_sizes=[args.kernel1_size, args.kernel2_size, args.kernel3_size],
            fc_sizes=[args.fc1_size, args.fc2_size],
            dropout=args.dropout,
            conv_type=args.conv_type
        )
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Move model to device
    model = model.to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1 and args.device == "cuda":
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    return model, device


def train_naive_bayes(data, args):
    """Train and evaluate Naive Bayes model."""
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    logging.info("Training Naive Bayes model...")
    start_time = time.time()
    
    model = train_nb(X_train, y_train)
    
    train_time = time.time() - start_time
    logging.info(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate on test set
    metrics = evaluate_nb(model, X_test, y_test)
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"naive_bayes_model.pkl")
    model.save(model_path)
    
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.model_dir, f"naive_bayes_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return model, metrics


def train_cnn(model, dataloaders, args, device):
    """Train CNN models with checkpoint saving and resume capability."""
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if args.model_type == "cnn_advanced":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
    
    # Checkpoint paths
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, f"{args.model_type}_checkpoint.pt")
    final_model_path = os.path.join(args.model_dir, f"{args.model_type}_model.pt")
    
    # Resume from checkpoint if requested and checkpoint exists
    start_epoch = 0
    best_val_loss = float('inf')
    history = None
    
    if args.resume and os.path.exists(checkpoint_path):
        model, start_epoch, best_val_loss, history = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        logging.info(f"Resuming from epoch {start_epoch}")
    
    # Create history dictionary if starting fresh
    if history is None:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    # Training loop
    logging.info(f"Training {args.model_type} model from epoch {start_epoch} to {args.epochs}...")
    
    if args.model_type == "cnn_advanced":
        model, history = train_advanced(
            model=model,
            train_loader=dataloaders['train_loader'],
            val_loader=dataloaders['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=args.epochs,
            device=device,
            model_dir=args.model_dir,
            early_stopping_patience=args.patience,
            mixed_precision=args.mixed_precision,
            checkpoint_callback=lambda model_state, epoch, val_loss, is_best: save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss,
                    'history': history
                },
                checkpoint_path,
                is_best
            ),
            start_epoch=start_epoch
        )
    else:  # cnn_standard
        model, history = train_standard(
            model=model,
            train_loader=dataloaders['train_loader'],
            val_loader=dataloaders['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=args.epochs,
            device=device,
            early_stopping_patience=args.patience,
            grad_clip_val=args.grad_clip,
            start_epoch=start_epoch,
            checkpoint_callback=lambda model_state, epoch, val_loss, is_best: save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss,
                    'history': history
                },
                checkpoint_path,
                is_best
            )
        )
    
    # Evaluate on test set
    logging.info("Evaluating model on test set...")
    if args.model_type == "cnn_advanced":
        metrics = evaluate_advanced(
            model=model,
            test_loader=dataloaders['test_loader'],
            criterion=criterion,
            device=device
        )
    else:  # cnn_standard
        metrics = evaluate_standard(
            model=model,
            test_loader=dataloaders['test_loader'],
            criterion=criterion,
            device=device
        )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics
    }, final_model_path)
    
    logging.info(f"Final model saved to {final_model_path}")
    logging.info(f"Test accuracy: {metrics['test_accuracy'] if 'test_accuracy' in metrics else metrics['accuracy']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.model_dir, f"{args.model_type}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return model, metrics


def train_cnn_advanced_hpo(model, dataloaders, args, device):
    """Train CNN Advanced with hyperparameter optimization."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    import yaml
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    def objective(trial):
        # Hyperparameters to tune
        conv_channels = [
            trial.suggest_int("conv_channel_1", 32, 128, step=16),
            trial.suggest_int("conv_channel_2", 64, 256, step=32),
            trial.suggest_int("conv_channel_3", 128, 512, step=64)
        ]
        
        kernel_sizes = [
            trial.suggest_int("kernel_size_1", 5, 25, step=2),
            trial.suggest_int("kernel_size_2", 3, 15, step=2),
            trial.suggest_int("kernel_size_3", 3, 9, step=2)
        ]
        
        fc_sizes = [
            trial.suggest_int("fc_size_1", 512, 2048, step=256),
            trial.suggest_int("fc_size_2", 256, 1024, step=128)
        ]
        
        dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2, step=0.05)
        
        # Select the convolution type
        conv_type = trial.suggest_categorical("conv_type", ["symmetric", "standard"])
        
        # Create a new model with trial hyperparameters
        model = AdvancedCNN(
            n_classes=dataloaders['n_classes'],
            seq_length=args.window_size,
            n_channels=5,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            fc_sizes=fc_sizes,
            dropout=dropout,
            conv_type=conv_type
        ).to(device)
        
        # If multi-GPU, wrap the model
        if torch.cuda.device_count() > 1 and args.device == "cuda":
            model = torch.nn.DataParallel(model)
        
        # Define loss function 
        class LabelSmoothingCrossEntropy(torch.nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
            
            def forward(self, x, target):
                log_probs = torch.nn.functional.log_softmax(x, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
                nll_loss = nll_loss.squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
                return loss.mean()
        
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        
        # Define optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        # Training
        n_epochs = 15  # Reduced epochs for hyperparameter tuning
        patience = 3  # Early stopping patience
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloaders['train_loader']):
                sequences = batch['sequence'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Report intermediate objective values for pruning
                if batch_idx % 10 == 0:
                    trial.report(loss.item(), epoch * len(dataloaders['train_loader']) + batch_idx)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloaders['val_loader']:
                    sequences = batch['sequence'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(dataloaders['val_loader'])
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_val_loss
    
    # Create study
    logging.info("Starting hyperparameter optimization...")
    
    # Create a study object and optimize the objective function
    study_name = f"cnn_advanced_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///{args.model_dir}/optuna_studies.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=30),
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.hpo_trials, timeout=args.hpo_timeout)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    logging.info(f"Best trial value: {best_value}")
    logging.info(f"Best parameters: {best_params}")
    
    # Save best parameters to a file
    params_path = os.path.join(args.model_dir, "best_hyperparameters.yaml")
    with open(params_path, "w") as f:
        yaml.dump(best_params, f)
    
    # Visualization if possible
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Make sure the figures directory exists
        os.makedirs("figures", exist_ok=True)
        
        # Plot optimization history
        history_fig = plot_optimization_history(study)
        history_fig.write_image("figures/optuna_history.png")
        
        # Plot parameter importances
        param_fig = plot_param_importances(study)
        param_fig.write_image("figures/optuna_param_importances.png")
    except:
        logging.warning("Optuna visualization failed. Continuing without plots.")
    
    # Train final model with best parameters
    logging.info("Training final model with best hyperparameters...")
    
    # Extract parameters
    conv_channels = [
        best_params["conv_channel_1"],
        best_params["conv_channel_2"],
        best_params["conv_channel_3"]
    ]
    
    kernel_sizes = [
        best_params["kernel_size_1"],
        best_params["kernel_size_2"],
        best_params["kernel_size_3"]
    ]
    
    fc_sizes = [
        best_params["fc_size_1"],
        best_params["fc_size_2"]
    ]
    
    dropout = best_params["dropout"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    smoothing = best_params["label_smoothing"]
    conv_type = best_params["conv_type"]
    
    # Create model with best parameters
    model = AdvancedCNN(
        n_classes=dataloaders['n_classes'],
        seq_length=args.window_size,
        n_channels=5,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        fc_sizes=fc_sizes,
        dropout=dropout,
        conv_type=conv_type
    ).to(device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and args.device == "cuda":
        model = torch.nn.DataParallel(model)
    
    # Train final model with standard training function
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay
    
    # Create a wrapper class for the custom loss
    class LabelSmoothingCrossEntropy(torch.nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
        
        def forward(self, x, target):
            log_probs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
    
    # Now use the standard training function with our custom loss
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train final model
    checkpoint_path = os.path.join(args.model_dir, f"{args.model_type}_checkpoint.pt")
    
    model, history = train_advanced(
        model=model,
        train_loader=dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=args.epochs,
        device=device,
        model_dir=args.model_dir,
        early_stopping_patience=args.patience,
        mixed_precision=args.mixed_precision,
        checkpoint_callback=lambda model_state, epoch, val_loss, is_best: save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': val_loss,
                'history': history
            },
            checkpoint_path,
            is_best
        ),
        start_epoch=0  # Start fresh with the optimized hyperparameters
    )
    
    # Evaluate on test set
    logging.info("Evaluating model on test set...")
    metrics = evaluate_advanced(
        model=model,
        test_loader=dataloaders['test_loader'],
        criterion=criterion,
        device=device
    )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f"{args.model_type}_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics,
        'hyperparameters': best_params
    }, final_model_path)
    
    logging.info(f"Final model saved to {final_model_path}")
    logging.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.model_dir, f"{args.model_type}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return model, metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models for malaria classification")
    
    # General settings
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["naive_bayes", "cnn_standard", "cnn_advanced"],
                       help="Type of model to train")
    
    parser.add_argument("--data_dir", type=str, default="data/split",
                       help="Directory containing split data")
    
    parser.add_argument("--model_dir", type=str, default="models",
                       help="Directory to save model checkpoints")
    
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save training logs")
    
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from checkpoint if available")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Data processing settings
    parser.add_argument("--window_size", type=int, default=1000,
                       help="Window size for genomic sequences")
    
    parser.add_argument("--stride", type=int, default=500,
                       help="Stride for genomic sequences")
    
    parser.add_argument("--cache_size", type=int, default=128,
                       help="Cache size for genomic sequences")
    
    parser.add_argument("--sample_limit", type=int, default=0,
                       help="Limit number of samples for testing (0 for no limit)")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs to train")
    
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    
    parser.add_argument("--patience", type=int, default=5,
                       help="Patience for early stopping")
    
    parser.add_argument("--grad_clip", type=float, default=1.0,
                       help="Gradient clipping value")
    
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for training")
    
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")
    
    # CNN advanced specific settings
    parser.add_argument("--conv1_channels", type=int, default=64,
                       help="Channels for first convolutional layer")
    
    parser.add_argument("--conv2_channels", type=int, default=128,
                       help="Channels for second convolutional layer")
    
    parser.add_argument("--conv3_channels", type=int, default=256,
                       help="Channels for third convolutional layer")
    
    parser.add_argument("--kernel1_size", type=int, default=15,
                       help="Kernel size for first convolutional layer")
    
    parser.add_argument("--kernel2_size", type=int, default=9,
                       help="Kernel size for second convolutional layer")
    
    parser.add_argument("--kernel3_size", type=int, default=5,
                       help="Kernel size for third convolutional layer")
    
    parser.add_argument("--fc1_size", type=int, default=1024,
                       help="Size of first fully connected layer")
    
    parser.add_argument("--fc2_size", type=int, default=512,
                       help="Size of second fully connected layer")
    
    parser.add_argument("--dropout", type=float, default=0.4,
                       help="Dropout rate")
    
    parser.add_argument("--conv_type", type=str, default="symmetric",
                       choices=["symmetric", "standard"],
                       help="Type of convolution to use")
    
    # Hyperparameter optimization settings
    parser.add_argument("--hpo", action="store_true",
                       help="Perform hyperparameter optimization (CNN Advanced only)")
    
    parser.add_argument("--hpo_trials", type=int, default=20,
                       help="Number of hyperparameter optimization trials")
    
    parser.add_argument("--hpo_timeout", type=int, default=36000,
                       help="Timeout for hyperparameter optimization in seconds")
    
    return parser.parse_args()


def main():
    """Main training script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.model_type)
    
    # Log arguments
    logging.info(f"Training {args.model_type} model with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    # Prepare data
    logging.info("Preparing data...")
    dataloaders = prepare_dataloader(args)
    
    if args.model_type == "naive_bayes":
        # Train and evaluate Naive Bayes
        model, metrics = train_naive_bayes(dataloaders, args)
    else:
        # Create model for CNNs
        if args.model_type in ["cnn_standard", "cnn_advanced"]:
            model, device = create_model(args, dataloaders['n_classes'])
            logging.info(f"Model created on {device}")
            
            # Log model summary
            try:
                from torchsummary import summary
                if args.model_type == "cnn_standard":
                    summary_input = (5, args.window_size)
                else:
                    summary_input = (args.window_size, 5)
                logging.info(f"Model summary:")
                logging.info(summary(model, summary_input))
            except:
                logging.warning("Could not log model summary. Install torchsummary for detailed model info.")
            
            # Train and evaluate advanced CNN with hyperparameter optimization
            if args.model_type == "cnn_advanced" and args.hpo:
                model, metrics = train_cnn_advanced_hpo(model, dataloaders, args, device)
            else:
                # Train and evaluate CNN standard or advanced without HPO
                model, metrics = train_cnn(model, dataloaders, args, device)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
    
    logging.info("Training completed!")
    return model, metrics


if __name__ == "__main__":
    main()
