import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self,
                 ckpt_folder: str,
                 patience: int = 7,
                 delta: float = 0.1):
        """
        Args:
            patience: How long to wait
                      after last time validation loss improved.
            delta   : minumum change in monitored quantify to quality
                      as an improvement
        """
        self.ckpt_folder = ckpt_folder
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss: float, model: nn.Module):
        score = - val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("Early stopping counter: {} out of {}".format(
                self.counter, self.patience
            ))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        torch.save(model.state_dict(), self.ckpt_folder + '/best_checkpoint.pt')
        self.val_loss_min = val_loss
