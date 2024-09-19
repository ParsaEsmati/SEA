import torch
import numpy as np
import os
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, DataLoader
from models.encoder_decoder import SpatialModel
from utils.modular_testing import unit_test_create_partitions2D, unit_test_create_partitions3D

class DataPartitioner2D:
    def __init__(self, x_coords, y_coords, vars, m=9, n=9, pad_id=-1, pad_field_value=0, device='cpu'):
        self.device = device
        self.x_coords = x_coords.to(self.device).float()
        self.y_coords = y_coords.to(self.device).float()
        self.full_coords = torch.stack((self.x_coords, self.y_coords), dim=1)

        self.var_list = [var.to(self.device).float() for var in vars if var is not None]

        if len(self.var_list) == 0:
            raise ValueError("At least one variable must be provided")

        self.m = m
        self.n = n
        self.pad_id = pad_id
        self.pad_field_value = pad_field_value

    def create_partitions(self):
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
        reconstructed_coords = torch.empty_like(self.full_coords)                  # [C,2]
        reconstructed_fields = torch.empty_like(torch.stack(self.var_list, dim=2)) # [B,C,F]

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.padded_partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions): # fields: [B,C,F]
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
        reconstructed_coords = torch.empty_like(self.full_coords)                  # [C,3]
        reconstructed_fields = torch.empty_like(torch.stack(self.var_list, dim=2)) # [B,C,F]

        time_dim   = time_dim            if time_dim            is not None else reconstructed_fields.shape[0]
        partitions = external_partitions if external_partitions is not None else self.partitions

        reconstructed_fields = reconstructed_fields[:time_dim]

        for idx, (coords, fields) in enumerate(partitions): # fields: [B,C,F]
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
        
# TODO: we have to add all the parameters to the temporal config 
class ProcessData:
    def __init__(self, n_inp, n_patches, model_path, spatial_model_dim, batch_size, config_spatial, device='cpu'):
        self.num_real_fields = 12
        self.num_fields = self.num_real_fields // 2
        self.num_embedded_fields = 2
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.n_inp = n_inp
        self.embed_dim = spatial_model_dim
        self.P = n_patches
        self.config_spatial = config_spatial
        self.model_spatial = self.initialize_spatial_model()
        self.load_model()

    def initialize_spatial_model(self):
        return SpatialModel(
            field_groups=self.config_spatial['field_groups'],
            n_inp=self.n_inp,
            MLP_hidden=self.n_inp * self.config_spatial['mlp_hidden_expansion_fact'],
            num_layers=self.config_spatial['num_layers'],
            embed_dim=self.config_spatial['embed_dim'],
            n_heads=self.config_spatial['n_heads'],
            max_len=self.config_spatial.get('max_len', 1000),
            src_len=self.config_spatial.get('src_len', 0),
            dropout=self.config_spatial['dropout']
        ).to(self.device)

    def load_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        self.model_spatial.load_state_dict(new_state_dict)
        self.model_spatial.eval()

    def initialize_and_process_data(self, data):
        data_spatial_dataset = EncoderDecoderDataset(data)
        dataloader = DataLoader(data_spatial_dataset, batch_size=self.batch_size, shuffle=False)
        processed_data = self.process_data(dataloader)
        self.clear_gpu_memory()
        return processed_data

    def process_data(self, dataloader):
        self.model_spatial.to(self.device)
        processed_chunks = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                data = self.model_spatial.generate_padding_mask(data)
                z, _, _ = self.model_spatial.encode(data)
                # z is already in the shape [B, P, 2, D], so no need to reshape
                processed_chunks.append(z.cpu())

        self.clear_gpu_memory()
        return torch.cat(processed_chunks, dim=0)

    def decode_data(self, data, device=None):
        device = device or self.device
        data = data.to(device)
        B, T, F, D = data.shape
        P = self.P
        data = data.reshape(B * T, P, F, D)  # Reshape to [B*T, P, 2, D]

        self.model_spatial.to(device)
        with torch.no_grad():
            decoded = self.model_spatial.decode(data)
            decoded = decoded.reshape(B, T, P, self.num_real_fields, -1)
            result = decoded.cpu()

        self.clear_gpu_memory()
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
        var_list = [scaled_data[:,:,i].to(torch.float32) for i in range(scaled_data.shape[-1])]
        
        m, n = self.config['m'], self.config['n']
        k = self.config['k'] if self.dimension == '3D' else None

        if self.dimension == '3D':
            self.partitioner = DataPartitioner3D(x_coords=self.coordinates[0], 
                                               y_coords=self.coordinates[1], 
                                               z_coords=self.coordinates[2],
                                               vars=var_list, m=m, n=n, k=k, 
                                               pad_id=-1, pad_field_value=0)
        else:
            self.partitioner = DataPartitioner2D(x_coords=self.coordinates[0], 
                                               y_coords=self.coordinates[1],
                                               vars=var_list, m=m, n=n, 
                                               pad_id=-1, pad_field_value=0)

        # Patchify the scaled data
        patched_data, _ = self.partitioner.create_partitions()
        
        if self.config.get('perform_initial_test', True):
            self._perform_initial_test(patched_data)

        fields_list = [part[1] for part in patched_data]
        coords_list = [part[0] for part in patched_data]

        stacked_fields = torch.stack(fields_list, dim=1)  # [T, P, C, F]
        self.stacked_coords = torch.stack(coords_list, dim=1)  # [T, P, C, 3] or [T, P, C, 2]

        return self.stacked_coords, stacked_fields

    def _scale_fields(self, fields: torch.Tensor) -> torch.Tensor:
        if self.scale_feature_range is None:
            return fields
        
        scaled_fields = torch.zeros_like(fields)
        for scaler, group in zip(self.scalers, self.field_groups):
            scaled_fields[..., group] = scaler.transform(fields[..., group])
        return scaled_fields

    def inverse_scale_and_unpatch(self, scaled_fields: torch.Tensor) -> torch.Tensor:  # [T, P, C, F]
        T, P, C, F = scaled_fields.shape
        unpatched_data = [(self.stacked_coords[:,i].to(torch.float32), scaled_fields[:,i].to(torch.float32)) for i in range(P)]
        reconstructed_coords, reconstructed_fields = self.partitioner.inverse_partition(unpatched_data, time_dim=T)

        # Inverse scale
        if self.scale_feature_range is not None:
            unscaled_fields = torch.zeros_like(reconstructed_fields)
            for scaler, group in zip(self.scalers, self.field_groups):
                unscaled_fields[..., group] = scaler.inverse_transform(reconstructed_fields[..., group])
        else:
            unscaled_fields = reconstructed_fields

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
