import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pprint


def CE_from_accuracy(accuracy: npt.NDArray, norm_accuracy: npt.NDArray):
    top_1_err = 1 - accuracy
    top_1_err_norm = 1 - norm_accuracy
    return top_1_err.sum() / top_1_err_norm.sum()


def RCE_from_accuracy(
    accuracy: npt.NDArray,
    norm_accuracy: npt.NDArray,
    clean_accuracy: float,
    norm_clean_accuracy: float,
):
    top_1_err = 1 - accuracy
    top_1_err_norm = 1 - norm_accuracy
    top_1_err_clean = 1 - clean_accuracy
    top_1_err_norm_clean = 1 - norm_clean_accuracy
    numerator = (top_1_err - top_1_err_clean).sum()
    denominator = (top_1_err_norm - top_1_err_norm_clean).sum()
    return numerator / denominator


def plot_part_a(args, severities):
    with open(args.input[0], "rb") as f:
        resnet18_res = pickle.load(f)
    plt.figure(figsize=(10, 5))
    colors = ["red", "green", "blue", "purple"]
    j = 0
    print(f"Plain test accuracy: {resnet18_res['plain']}")
    pprint.pprint(resnet18_res)
    for corruption in resnet18_res.keys():
        if corruption == "plain":
            continue
        plt.plot(
            severities,
            resnet18_res[corruption],
            label=f"{corruption}",
            color=colors[j],
            marker="o",
        )
        j += 1
    plt.axhline(resnet18_res["plain"], color="black", linestyle="dashed", label="plain")
    plt.xticks(severities)
    plt.xlabel("Corruption Severity")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.title(
        "ResNet18 Test accuracy vs corruption severity for various corruption functions"
        " applied to the CIFAR 10 Dataset"
    )
    plt.tight_layout()
    plt.show()


def plot_part_b(args):
    model_dict = {
        "debug": {},
        "resnet18": {},
        "vgg11": {},
        "vgg11_bn": {},
        "resnet34": {},
        "densenet121": {},
    }
    parsed_models = set()
    for path in args.input:
        with open(path, "rb") as f:
            name = path.split("-", 2)[1]
            parsed_models.add(name)
            data = pickle.load(f)
            # re-arrange data to give space for CE and RCE scores
            model_dict[name] = {k: {"test_accuracy": v} for k, v in data.items()}
    # parse the corruption functions and models to evaluate; sorting for determinism
    corr_funcs = sorted(
        [func for func in model_dict["resnet18"].keys() if func != "plain"]
    )
    ideal_order = ["vgg11", "vgg11_bn", "resnet34", "densenet121"]
    parsed_models = {
        model for model in parsed_models if model not in ["debug", "resnet18"]
    }
    eval_models = [model for model in ideal_order if model in parsed_models]
    # compute RCE and CE for each function for each model
    for model in eval_models:
        for func in corr_funcs:
            model_dict[model][func]["CE"] = CE_from_accuracy(
                model_dict[model][func]["test_accuracy"],
                model_dict["resnet18"][func]["test_accuracy"],
            )
            model_dict[model][func]["RCE"] = RCE_from_accuracy(
                model_dict[model][func]["test_accuracy"],
                model_dict["resnet18"][func]["test_accuracy"],
                model_dict[model]["plain"]["test_accuracy"],
                model_dict["resnet18"]["plain"]["test_accuracy"],
            )
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes = axes.flatten()
    bar_pos = np.arange(len(eval_models))
    bar_width = 0.35 / 2
    colors = plt.get_cmap("Blues_r")(np.linspace(0, 0.85, len(corr_funcs)))
    for score, ax in zip(["CE", "RCE"], axes):
        for i, (func, mult) in enumerate(zip(corr_funcs, [-3, -1, 1, 3])):
            model_scores = [model_dict[model][func][score] for model in eval_models]
            ax.barh(
                bar_pos + mult * (bar_width / 2),
                model_scores,
                bar_width,
                label=f"{func}",
                orientation="horizontal",
                color=colors[i],
            )
        ax.set_yticks(bar_pos)
        ax.set_yticklabels(eval_models)
        ax.set_xlabel(f"{score} value")
        ax.set_title(f"{score} score, normalized w.r.t. ResNet18")
    ax.legend()
    ax.invert_yaxis()
    fig.suptitle("CE and RCE scores of various models for various corruption functions")
    fig.set_tight_layout(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the necessary plots for 1.2")
    parser.add_argument("-i", "--input", help="Input file(s)", nargs="+", required=True)
    parser.add_argument("-p", "--part", type=str, default="a")

    args = parser.parse_args()

    severities = list(range(1, 6))
    if args.part == "a":
        plot_part_a(args, severities)
    elif args.part == "b":
        plot_part_b(args)
    else:
        raise ValueError("Invalid part")
