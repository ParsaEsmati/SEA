import torch.nn.functional as FF
import torch
import os
from abc import ABC, abstractmethod

class Vloss:
    def __init__(self, KL_weight_min, KL_weight_max, total_steps):
        self.recon_loss = None
        self.KL_loss = None
        self.total_loss = None
        self.KL_weight_min = KL_weight_min
        self.KL_weight_max = KL_weight_max
        self.total_steps = total_steps

    def __call__(self, x, z_mu, z_logvar, mu_recon, iteration, sigma_recon=None):
        x_recon = mu_recon
        self.KL_weight = self.KL_weight_min + (self.KL_weight_max - self.KL_weight_min) * (iteration / self.total_steps)
        self.recon_loss = FF.mse_loss(x_recon, x)
        self.KL_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        self.total_loss = self.recon_loss + self.KL_weight*self.KL_loss
        return self.total_loss
    

def initialize_optimizer(model, learning_rate=1e-4, total_steps=None): # config
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5) # config
    if total_steps is not None:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=total_steps) # config
        return optimizer, scheduler
    else:
        return optimizer


def calculate_R2(prediction, labels):
    prediction = prediction.reshape(-1)  # Flatten the tensor
    labels = labels.reshape(-1)  # Flatten the tensor
    residual = torch.sum((prediction - labels) ** 2)
    total = torch.sum((labels - torch.mean(labels)) ** 2)
    R2 = 1-(residual/total)
    return R2


class BaseErrorTracker(ABC):
    @abstractmethod
    def record_train_error(self, epoch, mse, R2, loss_fn):
        pass

    @abstractmethod
    def record_val_error(self, epoch, mse, R2, loss_fn):
        pass

    @abstractmethod
    def log_model(self, model, criterion, optimizer):
        pass

    @abstractmethod
    def finish(self):
        pass

class WandbErrorTracker(BaseErrorTracker):
    def __init__(self, project_name, run_name=None, config=None):
        import wandb
        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, name=run_name, config=config)

    def record_train_error(self, epoch, mse, R2, loss_fn):
        self.wandb.log({
            "epoch": epoch,
            "train/Recon": mse,
            "train/KL": loss_fn.KL_loss.item(),
            "train/R2": R2
        })

    def record_val_error(self, epoch, mse, R2, loss_fn):
        self.wandb.log({
            "epoch": epoch,
            "val/Recon": mse,
            "val/KL": loss_fn.KL_loss.item(),
            "val/R2": R2
        })

    def log_model(self, model, criterion, optimizer):
        self.wandb.watch(model, criterion, log="all", log_freq=10)

    def finish(self):
        self.wandb.finish()

class NoOpErrorTracker(BaseErrorTracker):
    def __init__(self, *args, **kwargs):
        pass

    def record_train_error(self, epoch, mse, R2, loss_fn):
        pass

    def record_val_error(self, epoch, mse, R2, loss_fn):
        pass

    def log_model(self, model, criterion, optimizer):
        pass

    def finish(self):
        pass

def create_error_tracker(use_wandb, project_name, run_name=None, config=None):
    if use_wandb:
        try:
            import wandb
            WANDB_API_KEY = "KEY"
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            wandb.login()
            print("Successfully logged in to Weights & Biases!")
            return WandbErrorTracker(project_name, run_name, config)
        except ImportError:
            print("Wandb not installed. Using NoOpErrorTracker.")
            return NoOpErrorTracker()
        except Exception as e:
            print(f"Error initializing Wandb: {str(e)}. Using NoOpErrorTracker.")
            return NoOpErrorTracker()
    else:
        return NoOpErrorTracker()