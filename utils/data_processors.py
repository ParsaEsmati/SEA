import torch
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from models.encoder_decoder import SpatialModel
from utils.modular_testing import unit_test_create_partitions2D, unit_test_create_partitions3D

class DataPartitioner2D:
    def __init__(self, x_coords, y_coords, m=9, n=9, pad_id=-1, pad_field_value=0, device='cpu'):
        self.device = device
        self.x_coords = x_coords.to(self.device).float()
        self.y_coords = y_coords.to(self.device).float()
        self.full_coords = torch.stack((self.x_coords, self.y_coords), dim=1)

        self.m = m
        self.n = n
        self.pad_id = pad_id
        self.pad_field_value = pad_field_value

    def create_partitions(self, vars):
        self.var_list = [var.to(self.device).float() for var in vars if var is not None]

        if len(self.var_list) == 0:
            raise ValueError("At least one variable must be provided")

        x_min, x_max = torch.min(self.x_coords), torch.max(self.x_coords)
        y_min, y_max = torch.min(self.y_coords), torch.max(self.y_coords)

        x_boundary = torch.linspace(x_min, x_max, self.m, device=self.device)
        y_boundary = torch.linspace(y_min, y_max, self.n, device=self.device)

        x_indices = torch.bucketize(self.x_coords, x_boundary, right=True)
        y_indices = torch.bucketize(self.y_coords, y_boundary, right=True)

        x_indices.clamp_(1, self.m - 1)
        y_indices.clamp_(1, self.n - 1)

        partitions = []
        index_map = []

        for i in range(1, self.m):
            for j in range(1, self.n):
                mask = (x_indices == i) & (y_indices == j)
                indices = mask.nonzero(as_tuple=False).view(-1)
                index_map.append(indices)

                if torch.any(mask):
                    partition_coords = torch.stack((self.x_coords[mask], self.y_coords[mask]), dim=1)
                    partition_fields = torch.stack([var[:, mask] for var in self.var_list], dim=2)
                else:
                    partition_coords = torch.empty((0, 2), dtype=torch.float32, device=self.device)
                    partition_fields = torch.empty((self.var_list[0].shape[0], 0, len(self.var_list)), dtype=torch.float32, device=self.device)

                partitions.append((partition_coords, partition_fields))

        self.index_map = index_map
        self.padded_partitions, self.padded_index_map = self.pad_partitions(partitions, index_map)
        return self.padded_partitions, self.padded_index_map

    def pad_partitions(self, partitions, index_map):
        max_len = max(coords.shape[0] for coords, _ in partitions)

        padded_partitions = []
        padded_index_map  = []
        num_vars = len(self.var_list)

        for (coords, fields), indices in zip(partitions, index_map):
            pad_size = max_len - coords.shape[0]

            if pad_size > 0:
                pad_coords = torch.full((pad_size, 2), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_coords = torch.cat((coords, pad_coords), dim=0)

                pad_fields = torch.full((self.var_list[0].shape[0], pad_size, num_vars), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_fields = torch.cat((fields, pad_fields), dim=1)

                pad_indices = torch.full((pad_size,), self.pad_id, dtype=torch.int64, device=self.device)
                padded_indices = torch.cat((indices, pad_indices), dim=0)
            else:
                padded_coords = coords
                padded_fields = fields
                padded_indices = indices

            padded_partitions.append((padded_coords, padded_fields))
            padded_index_map.append(padded_indices)

        return padded_partitions, padded_index_map

    def inverse_partition(self, external_partitions=None, time_dim=None):
        reconstructed_coords = torch.empty_like(self.full_coords)

        dummy_var = torch.stack(self.var_list, dim=2)
        _,C,F = dummy_var.shape
        B = external_partitions[0][1].shape[0]
        reconstructed_fields = torch.empty((B,C,F))

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.padded_partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions):
            indices         =   self.padded_index_map[idx]
            valid_mask      =   indices != self.pad_id
            valid_indices   =   indices[valid_mask]

            reconstructed_coords[valid_indices,:]     = coords[valid_mask]
            reconstructed_fields[:, valid_indices, :] = fields[:, valid_mask]

        return reconstructed_coords, reconstructed_fields


class DataPartitioner3D:
    def __init__(self, x_coords, y_coords, z_coords, vars, m=9, n=9, k=9, pad_id=-1, pad_field_value=0, device='cpu'):
        self.device = device
        self.x_coords = x_coords.to(self.device).float()
        self.y_coords = y_coords.to(self.device).float()
        self.z_coords = z_coords.to(self.device).float()
        self.full_coords = torch.stack((self.x_coords, self.y_coords, self.z_coords), dim=1)

        self.var_list = [var.to(self.device).float() for var in vars if var is not None]

        if len(self.var_list) == 0:
            raise ValueError("At least one variable must be provided")

        self.m = m
        self.n = n
        self.k = k
        self.pad_id = pad_id
        self.pad_field_value = pad_field_value
        
    def create_partitions(self):
        x_min, x_max = torch.min(self.x_coords), torch.max(self.x_coords)
        y_min, y_max = torch.min(self.y_coords), torch.max(self.y_coords)
        z_min, z_max = torch.min(self.z_coords), torch.max(self.z_coords)

        x_boundary = torch.linspace(x_min, x_max, self.m, device=self.device)
        y_boundary = torch.linspace(y_min, y_max, self.n, device=self.device)
        z_boundary = torch.linspace(z_min, z_max, self.k, device=self.device)

        x_indices = torch.bucketize(self.x_coords, x_boundary, right=True)
        y_indices = torch.bucketize(self.y_coords, y_boundary, right=True)
        z_indices = torch.bucketize(self.z_coords, z_boundary, right=True)

        x_indices.clamp_(1, self.m - 1)
        y_indices.clamp_(1, self.n - 1)
        z_indices.clamp_(1, self.k - 1)

        partitions = []
        index_map = []

        for i in range(1, self.m):
            for j in range(1, self.n):
                for k in range(1, self.k):
                    mask = (x_indices == i) & (y_indices == j) & (z_indices == k)
                    indices = mask.nonzero(as_tuple=False).view(-1)
                    index_map.append(indices)

                    if torch.any(mask):
                        partition_coords = torch.stack((self.x_coords[mask], self.y_coords[mask], self.z_coords[mask]), dim=1)
                        partition_fields = torch.stack([var[:, mask] for var in self.var_list], dim=2)
                    else:
                        partition_coords = torch.empty((0, 3), dtype=torch.float32, device=self.device)
                        partition_fields = torch.empty((self.var_list[0].shape[0], 0, len(self.var_list)), dtype=torch.float32, device=self.device)

                    partitions.append((partition_coords, partition_fields))

        self.index_map = index_map
        self.padded_partitions, self.padded_index_map = self.pad_partitions(partitions, index_map)
        return self.padded_partitions, self.padded_index_map

    def pad_partitions(self, partitions, index_map):
        max_len = max(coords.shape[0] for coords, _ in partitions)

        padded_partitions = []
        padded_index_map  = []
        num_vars = len(self.var_list)

        for (coords, fields), indices in zip(partitions, index_map):
            pad_size = max_len - coords.shape[0]

            if pad_size > 0:
                pad_coords = torch.full((pad_size, 3), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_coords = torch.cat((coords, pad_coords), dim=0)

                pad_fields = torch.full((self.var_list[0].shape[0], pad_size, num_vars), self.pad_field_value, dtype=torch.float32, device=self.device)
                padded_fields = torch.cat((fields, pad_fields), dim=1)

                pad_indices = torch.full((pad_size,), self.pad_id, dtype=torch.int64, device=self.device)
                padded_indices = torch.cat((indices, pad_indices), dim=0)
            else:
                padded_coords = coords
                padded_fields = fields
                padded_indices = indices

            padded_partitions.append((padded_coords, padded_fields))
            padded_index_map.append(padded_indices)

        return padded_partitions, padded_index_map

    def inverse_partition(self, external_partitions=None, time_dim=None):
        reconstructed_coords = torch.empty_like(self.full_coords)

        dummy_var = torch.stack(self.var_list, dim=2)
        _,C,F = dummy_var.shape
        B = external_partitions[0][1].shape[0] if external_partitions else dummy_var.shape[0]
        reconstructed_fields = torch.empty((B,C,F), device=self.device)

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.padded_partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions):
            indices         =   self.padded_index_map[idx]
            valid_mask      =   indices != self.pad_id
            valid_indices   =   indices[valid_mask]

            reconstructed_coords[valid_indices,:]     = coords[valid_mask]
            reconstructed_fields[:, valid_indices, :] = fields[:, valid_mask]

        return reconstructed_coords, reconstructed_fields

class MinMaxScaler:
    def __init__(self, feature_range=(-1, 1), name='scaler', save_dir='/content/drive/My Drive/MLEnhancedSM/'):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None
        self.name = name
        self.save_file = os.path.join(save_dir, f'{name}_min_max_values.pt')

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        self.min_val = torch.min(data)
        self.max_val = torch.max(data)
        print(f'for the {self.name} min and max are:')
        print(self.min_val)
        print(self.max_val)
        if self.min_val == self.max_val:
            raise ValueError("Data has zero variance")
        self._record_values()

    def transform(self, data):
        if not isinstance(data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        if self.min_val is None or self.max_val is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' with training data before using 'transform'.")
        std = (data - self.min_val) / (self.max_val - self.min_val)
        scaled_data = std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        if not isinstance(scaled_data, torch.Tensor):
            raise TypeError("Input data must be a torch.Tensor")
        if self.min_val is None or self.max_val is None:
            raise ValueError("The scaler has not been fitted yet.")

        print(f"Input scaled data range: {scaled_data.min().item()}, {scaled_data.max().item()}")

        std = (scaled_data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        print(f"Standardized data range: {std.min().item()}, {std.max().item()}")

        original_data = std * (self.max_val - self.min_val) + self.min_val
        print(f"Output original data range: {original_data.min().item()}, {original_data.max().item()}")

        print(f'Loaded min and max are: {self.min_val}, {self.max_val}')
        return original_data

    def _record_values(self):
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        torch.save({'min_val': self.min_val, 'max_val': self.max_val}, self.save_file)

    def load_values(self, path=None):
        load_file = path if path else self.save_file
        if os.path.exists(load_file):
            saved_data = torch.load(load_file)
            self.min_val = saved_data['min_val']
            self.max_val = saved_data['max_val']
            print(f'for the {self.name} loaded min and max are:')
            print(self.min_val)
            print(self.max_val)
        else:
            raise FileNotFoundError(f"No saved values found at {load_file}")
        
class ProcessData:
    def __init__(self, n_inp, config):
        self.config = config
        self.model_path = config['encoder_decoder_path']
        self.batch_size = config['spatial_batch_size']
        self.device = config['device']
        self.n_inp = n_inp
        self.embed_dim = config['embed_dim_spatial']

        if config['dimension'] == '3D':
            self.P = (config['m']-1) * (config['n']-1) * (config['k']-1)
        else:
            self.P = (config['m']-1) * (config['n']-1)

    def initialize_spatial_model(self):
        return SpatialModel(
            field_groups=self.config['field_groups'],
            n_inp=self.n_inp,
            MLP_hidden=self.config['MLP_hidden_spatial'],
            num_layers=self.config['num_layers_spatial'],
            embed_dim=self.config['embed_dim_spatial'],
            n_heads=self.config['n_heads_spatial'],
            max_len=self.config['block_size_spatial'],
            src_len=self.config['src_len_spatial'],
            variational=self.config['variational_spatial'],
            dropout=self.config['dropout_spatial']
        ).to(self.device)

    def load_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.model_spatial.load_state_dict(new_state_dict)
        self.model_spatial.eval()

    def initialize_and_process_data(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            dataloader = data
        else:
            data_spatial_dataset = EncoderDecoderDataset(data)
            dataloader = DataLoader(data_spatial_dataset, batch_size=1000, shuffle=False)
        
        processed_data = self.process_data(dataloader)
        return processed_data

    def process_data(self, dataloader):
        self.model_spatial = self.initialize_spatial_model()
        self.load_model()
        self.model_spatial.to(self.device)
        processed_chunks = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                data = self.model_spatial.generate_padding_mask(data)
                if self.config['variational_spatial']:
                    z, _, _ = self.model_spatial.encode(data)
                else:
                    z = self.model_spatial.encode(data)
                processed_chunks.append(z.cpu())

        #self.clear_gpu_memory()
        return torch.cat(processed_chunks, dim=0)

    def decode_data(self, data):
        self.model_spatial = self.initialize_spatial_model()
        self.load_model()
        self.model_spatial.to(self.device)
        data = data.to(self.device)
        with torch.no_grad():
            decoded = self.model_spatial.decode(data)
            result = decoded.cpu()

        return result

    def clear_gpu_memory(self):
        if self.device != 'cpu':
            torch.cuda.empty_cache()
            self.model_spatial.cpu()
            for param in self.model_spatial.parameters():
                param.data = param.data.cpu()
                if param.grad is not None:
                    param.grad.data = param.grad.data.cpu()
            print("GPU memory cleared")


class EncoderDecoderDataset(Dataset):
    def __init__(self, precomputed_data):
        self.data = precomputed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_tensor = self.data[idx]
        return data_tensor  # Return the data as both input and target

class TemporalDataset(Dataset):
    def __init__(self, data_list, data_list_original, field_ib, src_len=64, overlap=0, device='cpu', time_shifting_flag=False):
        self.device              =    device
        self.data_list           =    data_list
        self.data_list_original  =    data_list_original
        self.field_ib            =    field_ib
        self.src_len             =    src_len
        self.overlap             =    overlap
        self.step                =    src_len - overlap
        self.time_shifting_flag  =    time_shifting_flag

        self.num_samples = 0
        self.segment_samples = []

        # Calculate the number of valid src in the data
        # this is for the model that does not use tgt, instead plays with mask to reveal information
        for data in data_list:
            num_pairs = data.shape[0] // self.step
            self.segment_samples.append(num_pairs)
            self.num_samples += num_pairs

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        src: Time index fed to model
        tgt: Time index past src the model should estimate
        tgt_original: Original spatially unprocessed tgt
        field_ib_out: Field input/boundary
        """

        # Determine which data segment the index falls into
        segment_index = 0
        cumulative_samples = 0

        for i, samples in enumerate(self.segment_samples):
            cumulative_samples += samples
            if idx < cumulative_samples:
                segment_index = i
                break

        if idx < cumulative_samples:
            data_idx = idx - (cumulative_samples - self.segment_samples[segment_index])
        else:
            raise IndexError("Index out of range")


        if self.time_shifting_flag:
            shift_idx = np.random.randint(0, self.data_list[segment_index].shape[0] - self.step)
        else:
            shift_idx = 0
        active_data = self.data_list[segment_index]
        active_data_original = self.data_list_original[segment_index]
        active_field_ib = self.field_ib[segment_index]

        start_idx = data_idx * self.step
        end_idx = start_idx + self.src_len

        src = active_data[start_idx + shift_idx :  end_idx + shift_idx]
        tgt = active_data[start_idx+1+ shift_idx :  end_idx+1+ shift_idx]
        tgt_original = active_data_original[start_idx+1+ shift_idx : end_idx+1+ shift_idx]
        field_ib_out = active_field_ib[start_idx + shift_idx : end_idx + shift_idx]

        return src, tgt, tgt_original, field_ib_out

class MeshProcessor:
    def __init__(self, config: Dict[str, Any], coordinates: Tuple[torch.Tensor, ...]): # coordinates: [3, C]
        self.config = config
        self.dimension = config.get('dimension', '3D')

        if 'field_groups' not in config:
            raise ValueError("'field_groups' must be specified in the config dictionary")
        self.field_groups = config['field_groups']
        self.scale_feature_range = config.get('scale_feature_range')
        self.save_dir = config['save_dir']
        self.csv_scale_name = config.get('csv_scale_name', 'scaler')
        
        if self.dimension == '3D' and coordinates.shape[0] is None:
            raise ValueError("3D processing requires x, y, and z coordinates")
        elif self.dimension == '2D' and coordinates.shape[0] is None:
            raise ValueError("2D processing requires x and y coordinates")
        
        self.coordinates = coordinates
        
        self.scalers = []
        if self.scale_feature_range is not None:
            for i, group in enumerate(self.field_groups):
                scaler_config = {
                    'feature_range': self.scale_feature_range,
                    'name': f"{self.csv_scale_name}-group{i}",
                    'save_dir': self.save_dir
                }
                scaler = MinMaxScaler(scaler_config)
                self.scalers.append(scaler)

    def patchify_and_scale(self, data: torch.Tensor, train_indices: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...], Any]:
        T, N, F = data.shape
        batch_stacked_fields = []
        batch_stacked_coords = []

        # Scale the data before patchifying
        if self.scale_feature_range is not None:
            if train_indices is not None:
                for i, (scaler, group) in enumerate(zip(self.scalers, self.field_groups)):
                    scaler.fit(data[:, :, group])
            else:
                try:
                    for i, scaler in enumerate(self.scalers):
                        scaler_file = os.path.join(self.save_dir, f"{self.csv_scale_name}-group{i}_min_max_values.pt")
                        scaler.load_values(scaler_file)
                    print(f"Loaded scaler values from {self.save_dir}")
                except FileNotFoundError as e:
                    raise ValueError(f"No saved scaler values found and train_indices is None. Error: {str(e)}")

        scaled_data = self._scale_fields(data)
        
        m, n = self.config['m'], self.config['n']
        k = self.config['k'] if self.dimension == '3D' else None

        if self.dimension == '3D':
            self.partitioner = DataPartitioner3D(x_coords=self.coordinates[0], 
                                               y_coords=self.coordinates[1], 
                                               z_coords=self.coordinates[2],
                                               m=m, n=n, k=k, 
                                               pad_id=-1, pad_field_value=0)
        else:
            self.partitioner = DataPartitioner2D(x_coords=self.coordinates[0], 
                                               y_coords=self.coordinates[1],
                                               m=m, n=n, 
                                               pad_id=-1, pad_field_value=0)

        # Patchify the scaled data
        for idx in range(0, len(data), 2048):
            batch_data = scaled_data[idx:idx+2048]
            var_list = [batch_data[:,:,i].to(torch.float32) for i in range(batch_data.shape[-1])]
            patched_data, _ = self.partitioner.create_partitions(var_list)

            fields_list = [part[1] for part in patched_data]
            coords_list = [part[0] for part in patched_data]

            stacked_fields = torch.stack(fields_list, dim=1)  # [B, P, C, F]
            stacked_coords = torch.stack(coords_list, dim=1)  # [T, P, C, 3] or [T, P, C, 2]

            batch_stacked_fields.append(stacked_fields)
            #batch_stacked_coords.append(stacked_coords)

        if self.config.get('perform_initial_test', True):
            self._perform_initial_test(patched_data)

        # Concatenate all batches
        final_stacked_fields = torch.cat(batch_stacked_fields, dim=0)  # [T, P, C, F]
        self.stacked_coords = stacked_coords  # [T, P, C, 3] or [T, P, C, 2]

        return self.stacked_coords, final_stacked_fields

    def _scale_fields(self, fields: torch.Tensor) -> torch.Tensor:
        if self.scale_feature_range is None:
            return fields
        
        scaled_fields = torch.zeros_like(fields)
        for scaler, group in zip(self.scalers, self.field_groups):
            scaled_fields[..., group] = scaler.transform(fields[..., group])
        return scaled_fields

    def inverse_scale_and_unpatch(self, scaled_fields: torch.Tensor) -> torch.Tensor:  # [T, P, C, F]
        T, P, C, F = scaled_fields.shape
        final_reconstructed_fields = []
        for idx in range(0, T, 2048):
            coords_process = self.stacked_coords
            scaled_fields_process = scaled_fields[idx:idx+2048]
            unpatched_data = [(coords_process[:,i].to(torch.float32), scaled_fields_process[:,i].to(torch.float32)) for i in range(P)]
            reconstructed_coords, reconstructed_fields = self.partitioner.inverse_partition(unpatched_data, time_dim=T)
            final_reconstructed_fields.append(reconstructed_fields)

        final_reconstructed_fields = torch.cat(final_reconstructed_fields, dim=0)

        # Inverse scale
        if self.scale_feature_range is not None:
            unscaled_fields = torch.zeros_like(final_reconstructed_fields)
            for scaler, group in zip(self.scalers, self.field_groups):
                unscaled_fields[..., group] = scaler.inverse_transform(final_reconstructed_fields[..., group])
        else:
            unscaled_fields = final_reconstructed_fields

        return unscaled_fields
    
    def _perform_initial_test(self, patched_data: List[Tuple[torch.Tensor, torch.Tensor]]):
        reconstructed_coordinations, reconstructed_fields = self.partitioner.inverse_partition(external_partitions=patched_data)
        print('Results of simple initial test:')
        if self.dimension == '3D':
            unit_test_create_partitions3D(
                truth_fields=self.partitioner.var_list,
                coordx=self.partitioner.x_coords,
                coordy=self.partitioner.y_coords,
                coordz=self.partitioner.z_coords,
                inversed_fields=reconstructed_fields,
                inversed_coordx=reconstructed_coordinations[:, 0],
                inversed_coordy=reconstructed_coordinations[:, 1],
                inversed_coordz=reconstructed_coordinations[:, 2]
            )
        else:
            unit_test_create_partitions2D(
                truth_fields=self.partitioner.var_list,
                coordx=self.partitioner.x_coords,
                coordy=self.partitioner.y_coords,
                inversed_fields=reconstructed_fields,
                inversed_coordx=reconstructed_coordinations[:, 0],
                inversed_coordy=reconstructed_coordinations[:, 1]
            )

