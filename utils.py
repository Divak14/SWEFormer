import torch
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def scatter_kde(ax,x, y, colormap):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    sc = ax.scatter(x, y, c=z, s=18, cmap=colormap)
    return sc

def evaluate_and_plot_elev(model, dataloader, criterion, title_prefix='', xlim=None, ylim=None, density_plot=False, colormap='viridis', device = 'cuda'):
    model.eval()  # Turn on evaluation mode
    total_loss = 0.0
    outputs = []
    targets_list = []
    total_batches = len(dataloader)

    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch_idx, (data,cat, water,elev, targets) in enumerate(dataloader):
            data, cat,water,elev, targets = data.to(device), cat.to(device), water.to(device), elev.to(device), targets.to(device)
            mask = data == -999  # Creating the mask
            mask = mask.any(dim=2)  # Adjust as per your specific use case

            output = model(data, cat, water,elev, mask)  # Forward pass
            loss = criterion(output, targets)  # Compute loss

            total_loss += loss.item()  # Accumulate loss

            # Collect outputs and targets for plotting
            #print(output.shape)
            outputs.append(output.cpu().detach().numpy())
            targets_list.append(targets.cpu().detach().numpy())

    average_loss = total_loss / total_batches
    print(f'Average Loss ({title_prefix}): {average_loss:.7f}')

    # Flatten the outputs and targets lists
    outputs = np.concatenate(outputs)*100
    targets_list = np.concatenate(targets_list)*100

    # Calculate metrics
    mse = mean_squared_error(targets_list, outputs)
    rmse = np.sqrt(mse)
    r2,_ = pearsonr(targets_list, outputs)
    #r2 = r2_score(targets_list, outputs)
    bias = np.mean(outputs - targets_list)

    print(f'Mean Squared Error (MSE) ({title_prefix}): {mse:.7f}')
    print(f'Root Mean Squared Error (RMSE) ({title_prefix}): {rmse:.7f}')
    print(f'R-squared (R2) ({title_prefix}): {r2:.7f}')
    print(f'Bias ({title_prefix}): {bias:.7f}')

    return average_loss, outputs, targets_list
