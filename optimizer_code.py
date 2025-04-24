import torch
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn as nn

def setup_training(model, loss_type='mse', lr=0.00005, weight_decay=0.01, betas=(0.9, 0.999), quantiles=[0.5], 
                   warmup_steps=10, total_steps=100, gamma=0.95):
    """
    Setup the criterion, optimizer, and learning rate scheduler for training.

    Parameters:
        model: torch.nn.Module
            The model to be trained.
        loss_type: str
            Type of loss function to use ('mse' or 'quantile').
        lr: float
            Learning rate for the optimizer.
        weight_decay: float
            Weight decay for regularization.
        betas: tuple
            Betas for the AdamW optimizer.
        quantiles: list
            List of quantiles to be used in QuantileLoss (only used if loss_type is 'quantile').
        warmup_steps: int
            Number of warmup steps for learning rate scheduler.
        total_steps: int
            Total number of training steps (epochs).
        gamma: float
            Decay factor for the learning rate scheduler after warmup.
    
    Returns:
        criterion: Loss function (MSELoss or QuantileLoss)
        optimizer: Optimizer (AdamW)
        scheduler: Combined learning rate scheduler (SequentialLR)
    """
    
    # Select the appropriate loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'quantile':
        criterion = QuantileLoss(quantiles=quantiles)
    else:
        raise ValueError("Invalid loss_type. Choose either 'mse' or 'quantile'.")
    
    # Define the optimizer (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # Define the learning rate schedulers
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    return criterion, optimizer, scheduler