import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from tqdm import tqdm

from alg.vae_new import bayes_classifier, construct_optimizer
from attacks.black_box import sticker_attack
from utils.utils import load_data, load_params
from utils.visualisation import plot_images


def load_model(data_name, vae_type, checkpoint_index, device=None):
    """
    Load a trained model with given parameters.

    Args:
        data_name (str): Dataset name (e.g., 'mnist').
        vae_type (str): Model type ('A', 'B', ..., 'G').
        checkpoint_index (int): Index of the checkpoint to load.
        dimZ (int, optional): Latent dimension. Default is 64.
        dimH (int, optional): Hidden layer size. Default is 500.
        device (torch.device, optional): Device to load the model onto.

    Returns:
        encoder, generator: The loaded encoder and generator models.
    """
    if data_name == "mnist":
        if vae_type == "A":
            from models.conv_generator_mnist_A import Generator
        elif vae_type == "B":
            from models.conv_generator_mnist_B import Generator
        elif vae_type == "C":
            from models.conv_generator_mnist_C import Generator
        elif vae_type == "D":
            from models.conv_generator_mnist_D import Generator
        elif vae_type == "E":
            from models.conv_generator_mnist_E import Generator
        elif vae_type == "F":
            from models.conv_generator_mnist_F import Generator
        elif vae_type == "G":
            from models.conv_generator_mnist_G import Generator
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")
        from models.conv_encoder_mnist import GaussianConvEncoder as Encoder

        input_shape = (1, 28, 28)
        n_channel = 64
        dimZ = 64
        dimH = 500
        dimY = 10
    elif data_name == "cifar10" or data_name == "gtsrb":
        if vae_type == "A":
            from models.conv_generator_cifar10_A import Generator
        elif vae_type == "B":
            from models.conv_generator_cifar10_B import Generator
        elif vae_type == "C":
            from models.conv_generator_cifar10_C import Generator
        elif vae_type == "D":
            from models.conv_generator_cifar10_D import Generator
        elif vae_type == "E":
            from models.conv_generator_cifar10_E import Generator
        elif vae_type == "F":
            from models.conv_generator_cifar10_F import Generator
        elif vae_type == "G":
            from models.conv_generator_cifar10_G import Generator
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")
        from models.conv_encoder_cifar10 import GaussianConvEncoder as Encoder

        input_shape = (3, 32, 32)
        n_channel = 128
        dimZ = 128
        dimH = 1000
        dimY = 10
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    if data_name == "gtsrb":
        dimY = 43

    generator = Generator(input_shape, dimH, dimZ, dimY, n_channel, "sigmoid", "gen")
    encoder = Encoder(input_shape, dimH, dimZ, dimY, n_channel, "enc")

    path_name = f"{data_name}_conv_vae_{vae_type}_{dimZ}/"
    filename = f"save/{path_name}checkpoint"

    load_params((encoder, generator), filename, checkpoint_index)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    generator = generator.to(device)

    return encoder, generator


def perform_attacks(
    data_name, sticker_sizes, batch_size, save_dir="./results/", device=None
):
    """
    Perform FGSM attack on a given model, evaluate accuracy vs. epsilon, and save results.

    Args:
        data_name (str): Dataset name ('mnist').
        vae_type (str): Model type ('A', 'B', ..., 'G').
        checkpoint_index (int): Index of the checkpoint to load.
        epsilons (list): List of epsilon values for FGSM attack.
        save_dir (str): Directory to save results.
    """
    vae_types = ["A", "B", "C", "D", "E", "F", "G"]

    attack_methods = ["Sticker"]
    accuracies = {vae_type: {} for vae_type in vae_types}
    if os.path.exists(save_dir):
        warnings.warn(
            f"Results directory {save_dir} already exists and may be overwritten."
        )

    os.makedirs(save_dir, exist_ok=True)
    input_shape = (1, 28, 28) if data_name == "mnist" else (3, 32, 32)
    dimY = 10 if data_name != "gtsrb" else 43
    ll = "l2"
    K = 10
    _, test_dataset = load_data(data_name, path="./data", labels=None, conv=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    example_adversarial_images = {sticker_size: [] for sticker_size in sticker_sizes}
    n_rows = min(batch_size, 5)  # number of rows of images to plot

    for vae_type in vae_types:
        encoder, generator = load_model(data_name, vae_type, 0)
        encoder.eval()

        if vae_type == "A":
            dec = (generator.pyz_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_A as lowerbound
        elif vae_type == "B":
            dec = (generator.pzy_params, generator.pxzy_params)
            from alg.lowerbound_functions import lowerbound_B as lowerbound
        elif vae_type == "C":
            dec = (generator.pyzx_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_C as lowerbound
        elif vae_type == "D":
            dec = (generator.pyzx_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_D as lowerbound
        elif vae_type == "E":
            dec = (generator.pyz_params, generator.pzx_params)
            from alg.lowerbound_functions import lowerbound_E as lowerbound
        elif vae_type == "F":
            dec = (generator.pyz_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_F as lowerbound
        elif vae_type == "G":
            dec = (generator.pzy_params, generator.pxz_params)
            from alg.lowerbound_functions import lowerbound_G as lowerbound
        else:
            raise ValueError(f"Unknown VAE type: {vae_type}")

        enc_conv = encoder.encoder_conv
        enc_mlp = encoder.enc_mlp
        enc = (enc_conv, enc_mlp)
        X_ph = torch.zeros(1, *input_shape).to(next(encoder.parameters()).device)
        Y_ph = torch.zeros(1, dimY).to(next(encoder.parameters()).device)
        _, eval_fn = construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type)

        accuracies[vae_type] = {attack: [] for attack in attack_methods}

        for sticker_size in sticker_sizes:
            for attack in attack_methods:
                correct = 0
                total = 0

                for images, labels in tqdm(
                    test_loader, desc=f"{attack}, sticker size={sticker_size}"
                ):
                    images, labels = images.to(
                        next(encoder.parameters()).device
                    ), labels.to(next(encoder.parameters()).device)

                    if attack == "Sticker":
                        adv_images = sticker_attack(images, sticker_size)
                    else:
                        raise ValueError(f"Unsupported attack: {attack}")

                    if len(example_adversarial_images[sticker_size]) < n_rows:
                        example_adversarial_images[sticker_size].append(adv_images)

                    y_pred = bayes_classifier(
                        adv_images,
                        (enc_conv, enc_mlp),
                        dec,
                        ll,
                        dimY,
                        lowerbound=lowerbound,
                        K=10,
                        beta=1.0,
                    )
                    correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
                    total += labels.size(0)

                accuracies[vae_type][attack].append(correct / total)
                print(
                    f"Sticker size: {sticker_size}, VAE type: {vae_type}, Attack: {attack}, Accuracy: {correct / total}"
                )

    example_adversarial_images = {
        sticker_size: torch.cat(example_adversarial_images[sticker_size], dim=0)[
            :n_rows
        ]
        for sticker_size in sticker_sizes
    }
    example_adversarial_images = torch.cat(
        [example_adversarial_images[sticker_size] for sticker_size in sticker_sizes],
        dim=0,
    )

    with open(
        os.path.join(save_dir, f"{data_name}_accuracy_vs_sticker_size.json"), "w"
    ) as f:
        json.dump(accuracies, f)

    example_adversarial_images = np.array(example_adversarial_images)
    plot_images(
        example_adversarial_images,
        shape=(28, 28) if data_name == "mnist" else (32, 32, 3),
        path=str(save_dir),
        filename=f"_{data_name}_adversarial_images",
        n_rows=n_rows,
        color=True,
    )

    return accuracies


def plot_results(json_file, save_dir, data_name, sticker_sizes):
    with open(json_file, "r") as f:
        accuracies = json.load(f)
    vae_types = list(accuracies.keys())
    attack_methods = list(accuracies[vae_types[0]].keys())
    letter_to_title = {
        "A": "GFZ",
        "B": "GFY",
        "C": "GBZ",
        "D": "GBY",
        "E": "DFX",
        "F": "DFZ",
        "G": "DBX",
    }
    fig, axes = plt.subplots(len(attack_methods), 1, figsize=(4, 12))
    num_vae_types = len(vae_types)
    cmap = cm.get_cmap("rainbow", num_vae_types)
    for i, attack in enumerate(attack_methods):
        for j, vae_type in enumerate(vae_types):
            color = cmap(j)
            if len(attack_methods) == 1:
                axes.plot(
                    sticker_sizes,
                    accuracies[vae_type][attack],
                    marker="o",
                    label=letter_to_title[vae_type],
                    linewidth=2,
                    color=color,
                )
            else:
                axes[i].plot(
                    sticker_sizes,
                    accuracies[vae_type][attack],
                    marker="o",
                    label=letter_to_title[vae_type],
                    linewidth=2,
                    color=color,
                )
        if len(attack_methods) == 1:
            axes.set_title(f"{attack} victim acc")
            axes.set_xlabel("Sticker size")
            axes.grid(linestyle="--")
            axes.legend()
        else:
            axes[i].set_title(f"{attack} victim acc")
            axes[i].set_xlabel("Sticker size")
            axes[i].grid(linestyle="--")
            axes[i].legend()

    # Save the plot
    plt.tight_layout()
    filename = f"{data_name}__accuracy_vs_sticker_size_combined.png"
    plt.savefig(
        os.path.join(
            save_dir,
            filename,
        )
    )
    plt.close()
    print(
        "Saved combined accuracy plot for all attacks to",
        os.path.join(save_dir, filename),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform sticker attack on a trained VAE."
    )

    parser.add_argument(
        "--data_name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "gtsrb"],
        help="Dataset name (e.g., 'mnist').",
    )

    parser.add_argument(
        "--sticker_sizes",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="List of sticker size values for sticker attack.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluating the model.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results/", help="Directory to save results."
    )
    parser.add_argument("--compute", action="store_true", help="Compute the results.")
    parser.add_argument(
        "--plot", action="store_true", help="Plot the results from the JSON file."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="JSON file containing the results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load the model onto (e.g., 'cuda:0').",
    )

    args = parser.parse_args()
    if args.compute:
        results = perform_attacks(
            data_name=args.data_name,
            sticker_sizes=args.sticker_sizes,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            device=args.device,
        )
    if args.plot:
        if args.json_file is None:
            print("No JSON file specified.")
            args.json_file = os.path.join(
                args.save_dir, f"{args.data_name}_accuracy_vs_sticker_size.json"
            )
        plot_results(args.json_file, args.save_dir, args.data_name, args.sticker_sizes)
