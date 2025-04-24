import torch
import matplotlib.pyplot as plt
import numpy as np

def train_model_elev(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, epochs, device, early_stopping_patience=10, min_delta=0.0001, model_save_path='best_model.pth'):
    # Initialize variables for tracking training progress
    training_losses, validation_losses = [], []
    training_rmse, validation_rmse = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    # Loop over epochs
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        total_train_loss, total_train_rmse = 0., 0.

        for batch_idx, (data, cat, water, elev, targets) in enumerate(train_dataloader):
            data, cat, water,elev, targets = data.to(device), cat.to(device), water.to(device), elev.to(device), targets.to(device)
            mask = data == -999
            mask = mask.any(dim=2)

            optimizer.zero_grad()
            output = model(data, cat, water,elev, mask)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate RMSE (Root Mean Squared Error)
            rmse = torch.sqrt(torch.mean((output * 100 - targets * 100) ** 2))
            total_train_rmse += rmse.item()

        # Average training loss and RMSE for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_rmse = total_train_rmse / len(train_dataloader)
        training_losses.append(avg_train_loss)
        training_rmse.append(avg_train_rmse)

        # Validation phase
        model.eval()
        total_val_loss, total_val_rmse = 0., 0.

        with torch.no_grad():
            for batch_idx, (data, cat, water,elev, targets) in enumerate(val_dataloader):
                data, cat, water,elev, targets = data.to(device), cat.to(device), water.to(device), elev.to(device), targets.to(device)
                mask = data == -999
                mask = mask.any(dim=2)

                output = model(data, cat, water,elev, mask)
                loss = criterion(output, targets)
                total_val_loss += loss.item()

                # Calculate RMSE for validation
                rmse = torch.sqrt(torch.mean((output * 100 - targets * 100) ** 2))
                total_val_rmse += rmse.item()

        # Average validation loss and RMSE for the epoch
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_rmse = total_val_rmse / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        validation_rmse.append(avg_val_rmse)

        # Print epoch results
        print(f'| Epoch {epoch}/{epochs} | Training Loss: {avg_train_loss:.7f} | Validation Loss: {avg_val_loss:.7f} | '
              f'Training RMSE: {avg_train_rmse:.7f} | Validation RMSE: {avg_val_rmse:.7f}')

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        # Adjust learning rate
        scheduler.step()

    # Plot training and validation results
    #plot_training_validation_curves(training_losses, validation_losses, training_rmse, validation_rmse)
    return validation_rmse[-1]



def plot_training_validation_curves(training_losses, validation_losses, training_rmse, validation_rmse):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    axs[0].plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
    axs[0].plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()

    # Plot RMSE
    axs[1].plot(range(1, len(training_rmse) + 1), training_rmse, label='Training RMSE')
    axs[1].plot(range(1, len(validation_rmse) + 1), validation_rmse, label='Validation RMSE')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('RMSE')
    axs[1].set_title('Training and Validation RMSE')
    axs[1].legend()

    plt.tight_layout()
    plt.show()