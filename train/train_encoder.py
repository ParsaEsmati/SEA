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
import random

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

def get_datasets(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, MeshProcessor]:
    train_sources, val_sources, test_sources, mesh_processor = process_data(config)

    dataset_train = EncoderDecoderDataset(train_sources)
    dataset_validation = EncoderDecoderDataset(val_sources)
    dataset_test = EncoderDecoderDataset(test_sources)

    batch_size = config['batch_size']
    shuffle = True

    trainLoader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    validationLoader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return trainLoader, validationLoader, testLoader, mesh_processor

def process_data(config: Dict[str, Any]) -> Tuple[EncoderDecoderDataset, EncoderDecoderDataset, EncoderDecoderDataset, MeshProcessor]:
    """
    Load data, process it, create train/val/test splits, and apply MeshProcessor.
    """
    seed = config.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    print(f"Train length: {train_length}")
    print(f"Val length: {val_length}")
    print(f"Test length: {test_length}")

    config['train_size'] = train_length
    
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

    #---------------------------------------------------------------------------------------------------
    # This is a switch to allow mixing while keeping the exact structure and dimensions
    if config['SEA_isolate']:    
        train_sources_tokenized = scaled_fields[train_indices, :, :].permute(0, 1, 3, 2)
        validation_sources_tokenized = scaled_fields[val_indices, :, :].permute(0, 1, 3, 2)
        test_sources_tokenized = scaled_fields[test_indices, :, :].permute(0, 1, 3, 2)
    elif config['SEA_mixed']:   
        B, P, C, F = scaled_fields.shape
        train_sources_tokenized = scaled_fields[train_indices, :, :].reshape(-1,P,F,C)
        validation_sources_tokenized = scaled_fields[val_indices, :, :].reshape(-1,P,F,C)
        test_sources_tokenized = scaled_fields[test_indices, :, :].reshape(-1,P,F,C)
    else:
        assert False, "Invalid SEA data configuration"
    #---------------------------------------------------------------------------------------------------
    
    n_inp = train_sources_tokenized.shape[3]
    config['n_inp'] = n_inp

    if config.get('print_split_sizes', True):
        print(f'Train size: {train_sources_tokenized.shape[0]}')
        print(f'Validation size: {validation_sources_tokenized.shape[0]}')
        print(f'Test size: {test_sources_tokenized.shape[0]}')
        print(f'Train shape: {train_sources_tokenized.shape}')
        print(f'Validation shape: {validation_sources_tokenized.shape}')
        print(f'Test shape: {test_sources_tokenized.shape}')

    return train_sources_tokenized, validation_sources_tokenized, test_sources_tokenized, mesh_processor

def get_model(config: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    model = SpatialModel(
        field_groups=config['field_groups'],
        n_inp=config['n_inp'],  # Use the extracted n_inp here
        MLP_hidden=config['MLP_hidden'],
        num_layers=config['num_layers'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        max_len=config['block_size'],
        src_len=config.get('src_len', 0),
        variational=config['variational'],
        dropout=config['dropout']
    )

    if config.get('load_pretrained', False):
        model_path = config['pretrained_model_path']
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")

    model = model.to(device)
    optimizer = initialize_optimizer(model, config)

    if config['variational']:
        total_steps = round(config['epoch_num'] * config['train_size'] // config['batch_size'])
        loss_fn = Vloss(config['KL_weight_min'], config['KL_weight_max'], total_steps)
    else:
        loss_fn = torch.nn.MSELoss()

    return model, loss_fn, optimizer

def pre_train_encoder(config: Dict[str, Any]):
    device = torch.device(config['device'])
    trainLoader, validationLoader, testLoader, mesh_processor = get_datasets(config)
    model, loss_fn, optimizer = get_model(config, device)

    return model, optimizer, trainLoader, validationLoader, testLoader, loss_fn, mesh_processor


def train(config: Dict[str, Any], error_tracker):
    model, optimizer, trainLoader, validationLoader, testLoader, loss_fn, mesh_processor = pre_train_encoder(config)
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
            
            if config['variational']:
                outputs, mu, logvar = model(data)
                loss = loss_fn(x=data, z_mu=mu, z_logvar=logvar, mu_recon=outputs, sigma_recon=None, iteration=iter)
            else:
                outputs = model(data)
                loss = loss_fn(outputs, data)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if config['variational']:
                train_recon_loss += loss_fn.recon_loss.item()
                train_kl_loss += loss_fn.KL_loss.item()
            else:
                train_recon_loss += loss.item()
            train_r2_sum += calculate_R2(outputs.detach(), data.detach()).item()
            num_train_batches += 1
            iter += 1

        # Calculate average losses and R2 score
        train_loss /= num_train_batches
        train_recon_loss /= num_train_batches
        if config['variational']:
            train_kl_loss /= num_train_batches
        train_r2 = train_r2_sum / num_train_batches

        # Log train errors
        train_metrics = {
            "Loss": train_loss,
            "Recon_Loss": train_recon_loss,
            "R2": train_r2
        }
        if config['variational']:
            train_metrics["KL_Loss"] = train_kl_loss
        error_tracker.record_error("train", epoch, train_metrics)

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
                    if config['variational']:
                        v_outputs, v_mu, v_logvar = model(v_data)
                        v_loss = loss_fn(v_data, v_mu, v_logvar, v_outputs, sigma_recon=None, iteration=iter)
                    else:
                        v_outputs = model(v_data)
                        v_loss = loss_fn(v_outputs, v_data)

                    val_loss += v_loss.item()
                    if config['variational']:
                        val_recon_loss += loss_fn.recon_loss.item()
                        val_kl_loss += loss_fn.KL_loss.item()
                    else:
                        val_recon_loss += v_loss.item()
                    val_r2_sum += calculate_R2(v_outputs, v_data).item()
                    num_val_batches += 1

            # Calculate average validation losses and R2 score
            val_loss /= num_val_batches
            val_recon_loss /= num_val_batches
            if config['variational']:
                val_kl_loss /= num_val_batches
            val_r2 = val_r2_sum / num_val_batches

            # Log validation errors
            val_metrics = {
                "Loss": val_loss,
                "Recon_Loss": val_recon_loss,
                "R2": val_r2
            }
            if config['variational']:
                val_metrics["KL_Loss"] = val_kl_loss
            error_tracker.record_error("val", epoch, val_metrics)

            print(f"\nEpoch: {epoch}/{config['epoch_num']}")
            if config['variational']:
                print(f"Train - Total Loss: {train_loss:.8f}, Recon Loss: {train_recon_loss:.8f}, KL Loss: {train_kl_loss:.8f}, R^2: {train_r2:.8f}")
                print(f"Val   - Total Loss: {val_loss:.8f}, Recon Loss: {val_recon_loss:.8f}, KL Loss: {val_kl_loss:.8f}, R^2: {val_r2:.8f}")
            else:
                print(f"Train - Loss: {train_loss:.8f}, R^2: {train_r2:.8f}")
                print(f"Val   - Loss: {val_loss:.8f}, R^2: {val_r2:.8f}")

            # Check if current model is the best so far
            if val_recon_loss < prev_error:
                prev_error = val_recon_loss
                print("--- New Best Model Saved ---")
                model.to('cpu')
                model_path = f"{config['save_dir']}/encoder_decoder_{config['case_name']}_{config['run_name']}.pt"
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