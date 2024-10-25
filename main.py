import argparse
import sys
from pathlib import Path
import importlib

import torch
from train.train_encoder import train as train_encoder_decoder
from train.train_temporal import train as train_temporal
from utils.train_utils import create_error_tracker
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_config(flow_type, model_type):
    try:
        config_module = importlib.import_module(f"configs.{flow_type}")
        if model_type == "encoder":
            return config_module.get_config_spatial()
        elif model_type == "temporal":
            return config_module.get_config_temporal()
        else:
            print(f"Error: Unknown model type '{model_type}'")
            sys.exit(1)
    except ImportError:
        print(f"Error: Unknown flow type '{flow_type}'. Make sure the corresponding config module exists in the configs package.")
        sys.exit(1)
    except AttributeError:
        print(f"Error: The config module for '{flow_type}' does not have the required get_config_{model_type} function.")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or test SEA models")
    parser.add_argument("flow_type", help="Type of flow (e.g., cylinder_flow, multiphase_flow)")
    parser.add_argument("model_type", choices=["encoder", "temporal"], help="Type of model to train or test")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    parser.add_argument("--model_path", help="Path to the pre-trained model for testing", required=False)
    args = parser.parse_args()

    # Load configuration based on flow_type and model_type
    config = get_config(args.flow_type, args.model_type)

    # Initialize error tracker
    error_tracker = create_error_tracker(
        use_wandb=config.get('use_wandb', True),
        project_name=config['project_name'],
        run_name=f"{args.flow_type}_{args.model_type}_{config['case_name']}_{args.mode}",
        config=config
    )

    set_seed(config['random_seed'])

    if args.model_type == "encoder":
        if args.mode == "train":
            # Train the encoder-decoder model
            trained_model = train_encoder_decoder(config, error_tracker)
            if config['final_save']:
                save_path = Path(config['save_dir']) / f"final_model_encoder_{config['case_name']}_{config['run_name']}.pt"
                torch.save(trained_model.state_dict(), str(save_path))
                print(f"Encoder-Decoder training completed. Model saved to {save_path}")

        elif args.mode == "test":
            from utils.data_processors import ProcessData
            from train.train_encoder import get_model, get_datasets
            from utils.train_utils import test_encoder_decoder

            if args.model_path:
                config['encoder_decoder_path'] = args.model_path
            else:
                config['encoder_decoder_path'] = f"{config['save_dir']}/encoder_decoder_{config['case_name']}_{config['run_name']}.pt"
            
            print(f"Using pretrained encoder model: {config['encoder_decoder_path']}")
            _, validationLoader, testLoader, mesh_processor = get_datasets(config)
            processor = ProcessData(config['n_inp'], config)
            test_results = test_encoder_decoder(processor, validationLoader, mesh_processor, config)


    elif args.model_type == "temporal":
        device = torch.device(config['device'])
        
        if args.mode == "train":
            if args.model_path:
                config['load_pretrained'] = True
                config['pretrained_model_path'] = args.model_path
                print(f"Continuing training from model: {args.model_path}")
            
            model, _, _, _, _, _ = train_temporal(config, error_tracker)
            if config['final_save']:
                save_path = Path(config['save_dir']) / f"final_model_temporal_{config['case_name']}_{config['run_name']}.pt"
                torch.save(model.state_dict(), str(save_path))
                print(f"Temporal model training completed. Model saved to {save_path}")
        elif args.mode == "test":
            from train.train_temporal import get_model, get_datasets, full_autoregressive_evaluation
            
            # Use the model path from args if provided, otherwise use the one from config
            if args.model_path:
                config['pretrained_model_path'] = args.model_path
            else:
                config['pretrained_model_path'] = config['save_dir'] + '/' + f'temporal_{config["case_name"]}_run1.pt'
            
            print(f"Using pretrained model: {config['pretrained_model_path']}")
            
            # Load the pre-trained model
            config['load_pretrained'] = True
            model, loss_fn, _ = get_model(config, device)
            
            # Get test dataloader
            _, _, testLoader, mesh_processor, processor = get_datasets(config)
            
            # Run full autoregressive evaluation
            test_results = full_autoregressive_evaluation(model, testLoader, loss_fn, device, processor, mesh_processor, config, epoch=0, plot_traj=True)
            print("Test Results:")
            for key, value in test_results.items():
                print(f"{key}: {value}")
            
            # You can add more processing or saving of test results here
        else:
            print(f"Error: Unknown mode '{args.mode}'")
            sys.exit(1)
    else:
        print(f"Error: Unknown model type '{args.model_type}'")
        sys.exit(1)

if __name__ == "__main__":
    main()