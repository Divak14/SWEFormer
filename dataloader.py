import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder

# Function to standardize each sequence individually
def standardize_sequence(sequence):
    # Identify the valid data (non -999 values)
    valid_data = sequence[sequence != -999]

    # Compute mean and standard deviation on valid data only
    mean = np.mean(valid_data)
    std = np.std(valid_data)

    # If std is zero (i.e., all valid data points are the same), just subtract the mean
    if std == 0:
        standardized_seq = sequence - mean
    else:
        # Standardize the sequence
        standardized_seq = (sequence - mean) / std

    # Restore the -999 padding
    standardized_seq[sequence == -999] = -999

    return standardized_seq

# Standardize each sequence in X_tbh_tbv
def standardize_sequences(X):
    standardized_X = []
    for seq in X:
        standardized_seq = np.apply_along_axis(standardize_sequence, 0, seq)
        standardized_X.append(standardized_seq)
    return np.array(standardized_X)

def prepare_training_dataloader(X_tbh_tbv, X_air_temp, X_ground_temp, cat_inp, 
                                water_frac, peak_swe, train_ratio=0.8, batch_size=64, **kwargs):
    """
    Prepares PyTorch DataLoaders for training and testing by reshaping, standardizing, 
    and concatenating input features, encoding categorical inputs, and splitting 
    the dataset into training and test sets.

    Args:
        X_tbh_tbv (np.array): TBH/TBV input features divided by air temperature.
        X_air_temp (np.array): Air temperature data.
        X_ground_temp (np.array): Ground temperature data.
        cat_inp (list or np.array): Categorical input to be encoded.
        water_frac (np.array): Water fraction data.
        peak_swe (np.array): Peak SWE values (target 1).
        train_ratio (float): Proportion of data to be used for training (default is 0.8).
        batch_size (int): Batch size for DataLoader (default is 64).
        **kwargs: Additional keyword arguments (e.g., 'elev' for elevation data).
        
    Returns:
        DataLoader: Training data loader.
        DataLoader: Test data loader.
    """
    
    # Step 1: Reshape air and ground temperature data
    X_air_temp_reshaped = X_air_temp.reshape(X_air_temp.shape[0], X_air_temp.shape[1], 1)
    X_ground_temp_reshaped = X_ground_temp.reshape(X_ground_temp.shape[0], X_ground_temp.shape[1], 1)

    # Step 2: Concatenate the original TBH/TBV data with reshaped air and ground temperature data
    Final_train = np.concatenate([standardize_sequences(X_tbh_tbv), 
                                  X_air_temp_reshaped - 273.15,  # Convert to Celsius
                                  X_ground_temp_reshaped - 273.15], axis=2)
    
    X_tbh_tbv_standardized = Final_train

    # Step 3: Prepare the target values (Normalized peak SWE)
    y_peak_swe = peak_swe / 100  # Normalize peak SWE by dividing by 100
    y_train = y_peak_swe

    # Step 4: Convert input features and target to PyTorch tensors
    X = torch.tensor(X_tbh_tbv_standardized, dtype=torch.float32)  # Shape: (samples, seq_len, features)
    y = torch.tensor(y_train, dtype=torch.float32)  # Shape: (samples, 3)

    # Step 5: Encode the categorical input using LabelEncoder
    label_encoder = LabelEncoder()
    cat_inp_encoded = label_encoder.fit_transform(cat_inp)  # Convert to integer labels
    cat_inp_tensor = torch.tensor(cat_inp_encoded, dtype=torch.long)  # Use long dtype for integer labels
    
    # Step 6: Process additional data (elevation and water fraction)
    elev = kwargs.get('elev')  # Retrieve elevation if passed in kwargs
    water_frac_tensor = torch.tensor(np.sqrt(water_frac), dtype=torch.float32)
    elev_tensor = torch.tensor(np.sqrt(elev) / 100, dtype=torch.float32)  # Normalize elevation

    # Step 7: Stack data into a single TensorDataset
    dataset = TensorDataset(X, cat_inp_tensor, water_frac_tensor, elev_tensor, y)
    
    # Step 8: Calculate the lengths for train and test datasets
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Step 9: Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Step 10: Create DataLoaders for both train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return the DataLoaders
    return train_loader, test_loader


def prepare_test_dataloader(X_tbh_tbv_original, X_air_temp, X_ground_temp, cat_inp, water_frac, peak_swe,  batch_size=64,  **kwargs):
    
    
    
    X_air_temp_reshaped = X_air_temp.reshape(X_air_temp.shape[0], X_air_temp.shape[1], 1)
    X_ground_temp_reshaped = X_ground_temp.reshape(X_ground_temp.shape[0], X_ground_temp.shape[1], 1)

    # Concatenate the original tbh/tbv data with reshaped air and ground temperature data
    Final_train = np.concatenate([standardize_sequences(X_tbh_tbv_original), X_air_temp_reshaped-273.15, X_ground_temp_reshaped-273.15], axis=2)
    X_tbh_tbv_standardized = Final_train
    #X_tbh_tbv_standardized = standardize_sequences(Final_train)

    
    
    
    y_peak_swe = peak_swe / 100  # Normalized peak SWE
    y_train = y_peak_swe
        
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X_tbh_tbv_standardized, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    # Encode categorical input using LabelEncoder
    label_encoder = LabelEncoder()
    cat_inp_encoded = label_encoder.fit_transform(cat_inp)
    cat_inp_tensor = torch.tensor(cat_inp_encoded, dtype=torch.long)
    
    elev = kwargs.get('elev')
            
    water_frac_tensor = torch.tensor(np.sqrt(water_frac), dtype=torch.float32)
    elev_tensor = torch.tensor(np.sqrt(elev)/100, dtype=torch.float32)
    #water_elev = torch.stack((water_frac_tensor, elev_tensor), dim=1)
    dataset = TensorDataset(X, cat_inp_tensor, water_frac_tensor,elev_tensor, y)
           


    # Create DataLoader for the dataset
    test_loader_new = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return test_loader_new