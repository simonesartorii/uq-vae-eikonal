import os
import numpy as np

class Trainer:
    def __init__(self, output_dir: str, result_dir: str):
        self.output_dir = output_dir
        self.result_dir = result_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def _check_early_stopping(self, val_loss_val, best_val_loss, counter, patience):
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            counter = 0
            return best_val_loss, counter, True
        else:
            counter += 1
            return best_val_loss, counter, False

    def _weights_path(self, tag: str):
        path = os.path.join(self.output_dir, f'{tag}.weights.h5')
        return path
    
    def _history_path(self, tag: str):
        path = os.path.join(self.result_dir, f'{tag}.loss.npz')
        return path

    def save_weights_and_history(self, model, history_dict: dict, tag: str):
        weights_path = self._weights_path(tag)
        history_path = self._history_path(tag)

        model.save_weights(weights_path)
        np.savez(history_path, **history_dict)

        print(f"Weights saved in: {weights_path}")
        print(f"History saved in: {history_path}")
        
        return weights_path, history_path