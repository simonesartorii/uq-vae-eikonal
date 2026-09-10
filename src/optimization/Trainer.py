import os
import numpy as np
import json
from datetime import datetime

class Trainer:
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id

    def _check_early_stopping(self, val_loss_val, best_val_loss, counter, patience):
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            counter = 0
            return best_val_loss, counter, True
        else:
            counter += 1
            return best_val_loss, counter, False

    def _weights_path(self, tag: str):
            path = f'data/models/weights/{self.pipeline_id}_{tag}.weights.h5'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path
    
    def _history_path(self, tag: str):
            path = f'data/models/histories/{self.pipeline_id}_{tag}.loss.npz'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path

    def save_weights_and_history(self, model, history_dict: dict, tag: str):
        weights_path = self._weights_path(tag)
        history_path = self._history_path(tag)

        model.save_weights(weights_path)
        np.savez(history_path, **history_dict)

        print(f"Pesi salvati in: {weights_path}")
        print(f"History salvata in: {history_path}")
        
        return weights_path, history_path

    def update_experiment_log(self, phase_name: str, phase_data: dict, json_path: str = 'data/models/experiments_pipeline.json'):
        cartella = os.path.dirname(json_path)
        if cartella != "":
            os.makedirs(cartella, exist_ok=True)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        phase_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        esperimento_trovato = False
        
        for esperimento in logs:
            if esperimento.get("pipeline_id") == self.pipeline_id:
                esperimento[phase_name] = phase_data
                esperimento_trovato = True
                break

        if not esperimento_trovato:
            nuovo_esperimento = {
                "pipeline_id": self.pipeline_id,
                phase_name: phase_data
            }
            logs.append(nuovo_esperimento)

        with open(json_path, 'w') as f:
            json.dump(logs, f, indent=4)
            
        print(f"fase '{phase_name}' salvata.")