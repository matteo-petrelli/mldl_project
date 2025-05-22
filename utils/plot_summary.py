import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_summary(summary_csv="summary.csv", output_dir="experiments"):
    """Generate bar plot from summary.csv and save it."""
    df = pd.read_csv(summary_csv)

    # Sort experiments by validation accuracy
    df = df.sort_values(by="val_acc", ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    indices = range(len(df))

    plt.bar(indices, df["val_acc"], bar_width, label="Val Accuracy")
    plt.bar(
        [i + bar_width for i in indices],
        df["test_acc"],
        bar_width,
        label="Test Accuracy",
        alpha=0.7
    )

    plt.xlabel("Experiments")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation vs Test Accuracy by Hyperparameter Config")
    plt.xticks(
        [i + bar_width / 2 for i in indices],
        df["experiment"],
        rotation=45,
        ha="right"
    )
    plt.legend()
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "summary_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_summary()
