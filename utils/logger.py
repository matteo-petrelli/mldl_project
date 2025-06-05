import json
import os

class MetricLogger:
    """
    Utility class to log and save training/validation metrics.
    """

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.metrics = []

        # Create log directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def log(self, entry: dict):
        """
        Logs a dictionary of metrics (e.g., from a training epoch).
        """
        self.metrics.append(entry)
        self._save()

    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_all(self):
        return self.metrics
