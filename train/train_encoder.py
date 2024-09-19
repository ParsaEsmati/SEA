import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
from typing import Dict, Any, Tuple, Optional
from utils.data_processors import EncoderDecoderDataset, MeshProcessor
from utils.train_utils import Vloss, initialize_optimizer, calculate_R2
from models.encoder_decoder import SpatialModel
from utils.modular_testing import test_mesh_processor_2d, test_mesh_processor_3d
import time
import sys

"""
config = {

    # General
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'case_name': 'cylinder_flow',
    'save_dir': './checkpoints',

    # Data loading parameters (from previous response)
    'field_data_path': '/path/to/field_data.npy',
    'input_path': '/path/to/input_data.npy',
    'coordinates_path': '/path/to/coordinates.npy',


    # Data splitting parameters
    'train_fraction': 0.6,
    'val_fraction': 0.2,
    'random_seed': 42,  # for reproducibility in shuffling

    # Mesh processing parameters
    'mesh_processor': {
        # Add any specific parameters for MeshProcessor here
        # For example:
        'dimension': '3D',  # or '2D'
        'field_groups': [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        'scale_feature_range': None,
        'csv_scale_name': 'scaler',
        'm': 5,  # number of partitions in x direction
        'n': 5,  # number of partitions in y direction
        'k': None,  # number of partitions in z direction (for 3D)
        'pad_id': -1,
        'pad_field_value': 0,
    },

    # Model parameters
    'num_fields': 12,
    'MLP_ratio': 4,
    'num_layers': 12,
    'embed_dim': 128,
    'n_heads': 8,
    'block_size': 10,
    

    # Testing options
    'test_mesh_structure': False,
    'perform_initial_test': True,

    # Logging options
    'validation_interval': 1,
    'final_save': False,

    # Data parameters
    'batch_size': 128,

    # Training parameters
    'learning_rate': 1e-4,
    'KL_weight_min': 0.1,
    'KL_weight_max': 0.9,
    'epoch_num': 100,
    'use_wandb': True,
    'project_name': 'SEA_CylinderFlow',
    'run_name': 'encoder_decoder'
}

"""

def load_and_convert(config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    def load_single_file(path: str) -> torch.Tensor:
        if path.endswith('.npy'):
            data = np.load(path)
            if config.get('convert_to_torch', True):
                data = torch.from_numpy(data)
        elif path.endswith('.pt'):
            data = torch.load(path)
        else:
            raise ValueError(f"Unsupported file format for {path}. Only .npy and .pt are supported.")

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor.")

        return data.to(config['device'])

    # Load field data
    field_data_path = config['field_data_path']
    field_data = load_single_file(field_data_path)

    # Load coordinates
    coordinates_path = config['coordinates_path']
    coordinates = load_single_file(coordinates_path)

    # Load input data if path is provided in config
    input_data = None
    if 'input_path' in config and config['input_path'] is not None:
        input_data_path = config['input_path']
        input_data = load_single_file(input_data_path)

    return field_data, coordinates, input_data

def process_data(config: Dict[str, Any]) -> Tuple[EncoderDecoderDataset, EncoderDecoderDataset, EncoderDecoderDataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load data, process it, create train/val/test splits, and apply MeshProcessor.
    """
    # Load data
    field_data, coordinates, data_input = load_and_convert(config)
    print(f"Field data shape: {field_data.shape}")
    print(f"Input data shape: {data_input.shape}")

    tr, T, C, F = field_data.shape
    field_data = field_data.reshape(tr*T, C, F)

    # Create train/val/test splits
    train_fraction = config['train_fraction']
    val_fraction = config['val_fraction']
    total_samples = field_data.shape[0]
    
    np.random.seed(config.get('random_seed', 42))
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_length = int(np.round(total_samples * train_fraction))
    val_length = int(np.round(total_samples * val_fraction))
    test_length = total_samples - train_length - val_length
    
    train_indices = indices[:train_length]
    val_indices = indices[train_length:train_length + val_length]
    test_indices = indices[train_length + val_length:]

    # Apply MeshProcessor
    mesh_processor = MeshProcessor(config, coordinates)
    stacked_coords, scaled_fields = mesh_processor.patchify_and_scale(field_data, train_indices=train_indices)

    # Optionally test mesh structure

    if config['test_mesh_structure']:
        reconstructed = mesh_processor.inverse_scale_and_unpatch(scaled_fields)
        if config['dimension'] == '2D':
            test_results = test_mesh_processor_2d(field_data.float(), reconstructed, coordinates)
        else:
            test_results = test_mesh_processor_3d(field_data.float(), reconstructed, coordinates)
        print(test_results)

    # Prepare data for model
    train_sources_tokenized = scaled_fields[train_indices, :, :].permute(0, 1, 3, 2)
    validation_sources_tokenized = scaled_fields[val_indices, :, :].permute(0, 1, 3, 2)
    test_sources_tokenized = scaled_fields[test_indices, :, :].permute(0, 1, 3, 2)

    if config.get('print_split_sizes', True):
        print(f'Train size: {train_sources_tokenized.shape[0]}')
        print(f'Validation size: {validation_sources_tokenized.shape[0]}')
        print(f'Test size: {test_sources_tokenized.shape[0]}')
        print(f'Train shape: {train_sources_tokenized.shape}')
        print(f'Validation shape: {validation_sources_tokenized.shape}')
        print(f'Test shape: {test_sources_tokenized.shape}')

    dataset_train = EncoderDecoderDataset(train_sources_tokenized)
    dataset_validation = EncoderDecoderDataset(validation_sources_tokenized)
    dataset_test = EncoderDecoderDataset(test_sources_tokenized)

    return dataset_train, dataset_validation, dataset_test, stacked_coords, scaled_fields, data_input


def pre_train_encoder(config: Dict[str, Any]):
    dataset_train, dataset_validation, dataset_test, stacked_coords, scaled_fields, data_input = process_data(config)
    
    # Get a sample from the dataset to determine the shape
    sample_data = dataset_train[0]
    print(f"Sample data shape: {sample_data.shape}")
    _, _, n_inp = sample_data.shape
    
    shuffle = True
    batch_size = config['batch_size']
    trainLoader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    validationLoader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=shuffle)
    testLoader = DataLoader(dataset_test, batch_size=batch_size, shuffle=shuffle)

    model = SpatialModel(
        field_groups=config['field_groups'],
        n_inp=n_inp,  # Use the extracted n_inp here
        MLP_hidden=config['MLP_hidden'],
        num_layers=config['num_layers'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        max_len=config['block_size'],
        src_len=config.get('src_len', 0),
        dropout=config['dropout']
    )

    optimizer = initialize_optimizer(model, learning_rate=config['learning_rate'], total_steps=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_steps = round(config['epoch_num'] * len(trainLoader))
    loss_fn = Vloss(config['KL_weight_min'], config['KL_weight_max'], total_steps)

    return model, optimizer, trainLoader, validationLoader, testLoader, loss_fn
    

def train(config: Dict[str, Any], error_tracker):
    model, optimizer, trainLoader, validationLoader, testLoader, loss_fn = pre_train_encoder(config)
    device = torch.device(config['device'])
    model.to(device)
    model.train()
    
    start_time = time.time()
    prev_error = float('inf')
    iter = 0

    error_tracker.log_model(model, loss_fn, optimizer)

    for epoch in range(1, config['epoch_num'] + 1):
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_r2_sum = 0.0
        num_train_batches = 0

        for data in trainLoader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs, mu, logvar = model(data)
            loss = loss_fn(x=data, z_mu=mu, z_logvar=logvar, mu_recon=outputs, sigma_recon=None, iteration=iter)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += loss_fn.recon_loss.item()
            train_kl_loss += loss_fn.KL_loss.item()
            train_r2_sum += calculate_R2(outputs.detach(), data.detach()).item()
            num_train_batches += 1
            iter += 1

        # Calculate average losses and R2 score
        train_loss /= num_train_batches
        train_recon_loss /= num_train_batches
        train_kl_loss /= num_train_batches
        train_r2 = train_r2_sum / num_train_batches

        # Log train errors
        error_tracker.record_train_error(epoch, train_recon_loss, train_r2, loss_fn)

        if epoch % config.get('validation_interval', 1) == 0 or epoch == config['epoch_num']:
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_r2_sum = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for v_data in validationLoader:
                    v_data = v_data.to(device)
                    v_outputs, v_mu, v_logvar = model(v_data)
                    v_loss = loss_fn(v_data, v_mu, v_logvar, v_outputs, sigma_recon=None, iteration=iter)

                    val_loss += v_loss.item()
                    val_recon_loss += loss_fn.recon_loss.item()
                    val_kl_loss += loss_fn.KL_loss.item()
                    val_r2_sum += calculate_R2(v_outputs, v_data).item()
                    num_val_batches += 1

            # Calculate average validation losses and R2 score
            val_loss /= num_val_batches
            val_recon_loss /= num_val_batches
            val_kl_loss /= num_val_batches
            val_r2 = val_r2_sum / num_val_batches

            # Log validation errors
            error_tracker.record_val_error(epoch, val_recon_loss, val_r2, loss_fn)

            print(f"\nEpoch: {epoch}/{config['epoch_num']}")
            print(f"Train - Total Loss: {train_loss:.8f}, Recon Loss: {train_recon_loss:.8f}, KL Loss: {train_kl_loss:.8f}, R^2: {train_r2:.8f}")
            print(f"Val   - Total Loss: {val_loss:.8f}, Recon Loss: {val_recon_loss:.8f}, KL Loss: {val_kl_loss:.8f}, R^2: {val_r2:.8f}")

            # Check if current model is the best so far
            if val_recon_loss < prev_error:
                prev_error = val_recon_loss
                print("--- New Best Model Saved ---")
                model.to('cpu')
                model_path = f"{config['save_dir']}/encoder_decoder_{config['case_name']}_{config['run_name']}_best.pt"
                torch.save(model.state_dict(), model_path)
                model.to(device)
            else:
                print("--- No Improvement, Best Model Retained ---")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")

    # Finish error tracking
    error_tracker.finish()

    return model