import torch
def get_config_spatial():
    config = {
        # General
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints',
        'field_data_path': '/home/parsa/projects/MLAcceleratedNumericalSim/SEA:StateExchangeAttention/data/CF/all_data/field_data.npy',
        'input_path': '/home/parsa/projects/MLAcceleratedNumericalSim/SEA:StateExchangeAttention/data/CF/all_data/input_data.npy',
        'coordinates_path': '/home/parsa/projects/MLAcceleratedNumericalSim/SEA:StateExchangeAttention/data/CF/all_data/coordinates.npy',

        # Data splitting parameters
        'train_fraction': 0.8,
        'val_fraction': 0.1,
        'random_seed': 42,
        # Mesh processing parameters
        'dimension': '2D',
        'field_groups': [[0, 1], [2]],
        'scale_feature_range': None,
        'csv_scale_name': 'scaler',
        'm': 9,
        'n': 9,
        'k': None,
        'pad_id': -1,
        'pad_field_value': 0,
        # Model parameters
        'MLP_hidden': 624,
        'num_layers': 12,
        'embed_dim': 32,
        'n_heads': 8,
        'block_size': 2024,
        'src_len': 0,
        'dropout': 0.0,
        'variational': False,
        # Testing options
        'test_mesh_structure': False,
        'perform_initial_test': True,
        # Logging options
        'validation_interval': 10,
        'final_save': False,
        # Data parameters
        'batch_size': 128,
        # Training parameters
        'learning_rate': 1e-4,
        'KL_weight_min': 0,
        'KL_weight_max': 0,
        'epoch_num': 5000,
        # wandb parameters
        'use_wandb': True,
        'run_name': 'run1',
        'case_name': 'multiphase_flow',
        'project_name': 'SEA_Encoder_Decoder',
        'WANDB_API_KEY': 'a408bff5577f4308fdd2594220f2119323956157',
        # For testing
        'spatial_batch_size': 1000,

        # SEA data configuration
        'SEA_isolate': True,
        'SEA_mixed': False
    }
    
    # Set embed_dim_spatial equal to embed_dim
    config['embed_dim_spatial'] = config['embed_dim']
    config['n_heads_spatial'] = config['n_heads']
    config['block_size_spatial'] = config['block_size']
    config['dropout_spatial'] = config['dropout']
    config['MLP_hidden_spatial'] = config['MLP_hidden']
    config['num_layers_spatial'] = config['num_layers']
    config['src_len_spatial'] = config['src_len']
    config['variational_spatial'] = config['variational']
    
    return config

def get_config_temporal():
    spatial_config = get_config_spatial()
    return {
        'device': spatial_config['device'],
        'save_dir': spatial_config['save_dir'],
        'field_data_path': spatial_config['field_data_path'],
        'input_path': spatial_config['input_path'],
        'coordinates_path': spatial_config['coordinates_path'],

        # Data splitting parameters
        'train_fraction': 0.6,
        'val_fraction': 0.2,
        'random_seed': 42,  # for reproducibility in shuffling

        # Mesh processing parameters
        'dimension': spatial_config['dimension'],
        'field_groups': spatial_config['field_groups'],
        'scale_feature_range': spatial_config['scale_feature_range'],
        'csv_scale_name': spatial_config['csv_scale_name'],
        'm': spatial_config['m'],
        'n': spatial_config['n'],
        'k': spatial_config['k'],
        'pad_id': spatial_config['pad_id'],
        'pad_field_value': spatial_config['pad_field_value'],

        # Spatial model parameters (Inference)
        'MLP_hidden_spatial': spatial_config['MLP_hidden'],
        'num_layers_spatial': spatial_config['num_layers'],
        'embed_dim_spatial': spatial_config['embed_dim'],
        'n_heads_spatial': spatial_config['n_heads'],
        'block_size_spatial': spatial_config['block_size'],
        'dropout_spatial': spatial_config['dropout'],
        'variational_spatial': spatial_config['variational'],
        'src_len_spatial': spatial_config['src_len'],
        'encoder_decoder_path': f"{spatial_config['save_dir']}/encoder_decoder_{spatial_config['case_name']}_{spatial_config['run_name']}.pt",
        'spatial_batch_size': spatial_config['batch_size'],
        

        # Temporal model parameters
        'num_layers': 1,
        'embed_dim': 2048,
        'n_heads': 8,
        'block_size': 2024,
        'scale_ratio': 8,
        'src_len': 0,
        'num_fields': len(spatial_config['field_groups']),
        'down_proj': 2,
        'dropout': 0.0,
        'exchange_mode': 'sea',
        'pos_encoding_mode': 'learnable',
        'ib_scale_mode': 'mlp',
        'ib_addition_mode': 'add',
        'ib_mlp_layers': 1, 
        'ib_num': 1, # Number of input/boundary provided to the model
        'add_info_after_cross': True,
        'LN_type': 'ln',

        # Testing options
        'test_mesh_structure': False,
        'perform_initial_test': True,

        # Logging options
        'validation_interval': 10,
        'full_eval_interval': 100,
        'final_save': False,

        # Data parameters
        'batch_size': 4,
        'dataset_src_len': 199,
        'dataset_overlap': 0,
        'dataset_time_shifting_flag': False,

        # Training parameters
        'variational': False,
        'learning_rate': 8e-5,
        'KL_weight_min': 0,
        'KL_weight_max': 0,
        'epoch_num': 3000,

        # wandb parameters
        'use_wandb': False,
        'run_name': 'run1',
        'case_name': 'cylinder_flow',
        'project_name': f'SEA_Temporal',
        'WANDB_API_KEY': None,

        # SEA data configuration
        'SEA_isolate': spatial_config['SEA_isolate'],
        'SEA_mixed': spatial_config['SEA_mixed']
    }

        