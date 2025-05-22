import os
import csv
import pandas as pd

def summarize_experiments(base_dir="experiments", output_file="summary.csv"):
    """Parse all experiment folders and create a summary CSV."""
    results = []

    for folder in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, folder)
        metrics_file = os.path.join(exp_path, "metrics.csv")

        if os.path.isfile(metrics_file):
            with open(metrics_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                best_val_acc = max(rows, key=lambda x: float(x["val_acc"]))
                best_test_acc = best_val_acc["test_acc"]

                results.append({
                    "experiment": folder,
                    "epoch": best_val_acc["epoch"],
                    "val_acc": float(best_val_acc["val_acc"]),
                    "test_acc": float(best_test_acc)
                })

    # Sort by validation accuracy descending
    results.sort(key=lambda x: x["val_acc"], reverse=True)

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"✅ Summary saved to {output_file}")
    print(df)

if __name__ == "__main__":
    summarize_experiments()
