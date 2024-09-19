import torch
def get_config():
    return {
    # General
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'case_name': 'cylinder_flow',
    'project_name': f'SEA_cylinder_flow',
    'save_dir': './checkpoints',

    # Data loading parameters (from previous response)
    'field_data_path': '/home/parsa/projects/MLAcceleratedNumericalSim/SEA:StateExchangeAttention/data/CF/field_data.npy', # TODO This should become relative if possible or user input
    'input_path': '/home/parsa/projects/MLAcceleratedNumericalSim/SEA:StateExchangeAttention/data/CF/input_data.npy',
    'coordinates_path': '/home/parsa/projects/MLAcceleratedNumericalSim/SEA:StateExchangeAttention/data/CF/coordinates.npy',


    # Data splitting parameters
    'train_fraction': 0.6,
    'val_fraction': 0.2,
    'random_seed': 42,  # for reproducibility in shuffling

    # Mesh processing parameters

    'dimension': '2D',  # or '2D'
    'field_groups': [[0, 1], [2]],
    'scale_feature_range': None,
    'csv_scale_name': 'scaler',
    'm': 5,  # number of partitions in x direction
    'n': 5,  # number of partitions in y direction
    'k': None,  # number of partitions in z direction (for 3D)
    'pad_id': -1,
    'pad_field_value': 0,


    # Model parameters
    'num_fields': 12,
    'MLP_hidden': 480,
    'num_layers': 12,
    'embed_dim': 128,
    'n_heads': 8,
    'block_size': 10,
    'dropout': 0.1,
    

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
    'KL_weight_min': 0,
    'KL_weight_max': 0,
    'epoch_num': 100,
    'use_wandb': True,
    'project_name': 'SEA_CylinderFlow',
    'run_name': 'encoder_decoder'
}

