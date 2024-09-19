import argparse
import sys
from pathlib import Path

import torch
from configs import cylinder_flow
from train.train_encoder import train as train_encoder_decoder
from utils.train_utils import create_error_tracker

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train SEA models")
    parser.add_argument("model_type", choices=["encoder", "temporal"], help="Type of model to train")
    args = parser.parse_args()

    # Load configuration
    config = cylinder_flow.get_config()

    # Initialize error tracker
    error_tracker = create_error_tracker(
        use_wandb=config.get('use_wandb', True),
        project_name=config['project_name'],
        run_name=f"{args.model_type}_{config['run_name']}",
        config=config
    )

    if args.model_type == "encoder":
        # Train the encoder-decoder model
        trained_model = train_encoder_decoder(config, error_tracker)
        
        if config['final_save']:
            save_path = Path(config['save_dir']) / f"final_model_encoder_{config['case_name']}_{config['run_name']}.pt"
            torch.save(trained_model.state_dict(), str(save_path))

        print(f"Encoder-Decoder training completed. Model saved to {save_path}")


    elif args.model_type == "temporal":
        print("Temporal model training not implemented yet.")
        sys.exit(0)

    else:
        print(f"Error: Unknown model type '{args.model_type}'")
        sys.exit(1)

if __name__ == "__main__":
    main()