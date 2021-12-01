import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np


def main(**kwargs):
    which_model = kwargs.pop("model")
    # read the data
    with open(f"{kwargs['results_dir']}{which_model}_results.pkl", "rb") as f:
        logging_info = pickle.load(f)
    # parse
    test_loss = logging_info["loss"]["test"]["regular"]
    permuted_test_loss = logging_info["loss"]["test"]["permuted"]
    val_losses = logging_info["loss"]["val"]
    # report metrics
    print(f"Test Loss:{test_loss}")
    print(f"Permuted Test Loss:{permuted_test_loss}")
    print(f"Validation Losses:{val_losses}")
    # plot
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, len(val_losses) + 1)
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", color='black')
    plt.ylabel("Average Epoch Loss")
    plt.xlabel("Epoch Number")
    plt.legend()
    plt.title(f"Validation Loss of {'MLP' if which_model =='mlp' else 'GNN'} model")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="mlp",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--results_dir",
        "-rd",
        default="./",
        type=str,
        help="Path to directory containing the "
        "results.pkl file, including final forward slash",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
