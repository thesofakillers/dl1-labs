import pickle
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the results of the training.")

    parser.add_argument(
        "--path",
        type=str,
        default="output/torch_10_nobn.pkl",
        help="Path to the results dictionary.",
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="whether the plots are for numpy curves",
        default=False,
    )
    args = parser.parse_args()
    with open(args.path, "rb") as f:
        logging_dict = pickle.load(f)

    train_loss = logging_dict["loss"]["train"]
    val_loss = logging_dict["loss"]["validation"]

    train_acc = logging_dict["accuracy"]["train"]
    val_acc = logging_dict["accuracy"]["validation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Loss Curves")
    ax1.plot(train_loss, label="training")
    ax1.plot(val_loss, label="validation")
    ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")

    ax2.set_title("Accuracy Curves")
    ax2.plot(train_acc, label="training")
    ax2.plot(val_acc, label="validation")
    ax2.legend()
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    fig.suptitle(f"{'NumPy' if args.numpy else 'PyTorch'} training curves")

    fig.set_tight_layout(True)
    plt.show()
