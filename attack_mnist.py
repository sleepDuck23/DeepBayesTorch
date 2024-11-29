import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from tqdm import tqdm

from alg.vae_new import bayes_classifier, construct_optimizer
from utils.utils import load_data, load_params
from utils.visualisation import plot_images


def load_model(data_name, vae_type, checkpoint_index, dimZ=64, dimH=500, device=None):
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

    # Instantiate the models
    generator = Generator(input_shape, dimH, dimZ, 10, n_channel, "sigmoid", "gen")
    encoder = Encoder(input_shape, dimH, dimZ, 10, n_channel, "enc")

    # Build checkpoint path
    path_name = f"{data_name}_conv_vae_{vae_type}_{dimZ}/"
    filename = f"save/{path_name}checkpoint"

    # Load model parameters
    load_params((encoder, generator), filename, checkpoint_index)

    # Move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    generator = generator.to(device)

    return encoder, generator


def perform_fgsm_attack(
    data_name, vae_type, checkpoint_index, epsilons, save_dir="./results/"
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
    # Load model
    encoder, generator = load_model(data_name, vae_type, checkpoint_index)
    encoder.eval()
    input_shape = (1, 28, 28)
    n_channel = 64
    dimY = 10
    ll = "bernoulli"  # Adjust based on likelihood used in training
    K = 1

    # Define decoder components
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

    # Construct optimizer functions
    enc_conv = encoder.encoder_conv
    enc_mlp = encoder.enc_mlp
    enc = (enc_conv, enc_mlp)
    X_ph = torch.zeros(1, *input_shape).to(next(encoder.parameters()).device)
    Y_ph = torch.zeros(1, dimY).to(next(encoder.parameters()).device)
    _, eval_fn = construct_optimizer(X_ph, Y_ph, enc, dec, ll, K, vae_type)

    # Prepare test dataset
    _, test_dataset = load_data(data_name, path="./data", labels=None, conv=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    # Ensure results directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Store accuracy results
    accuracies = []

    for epsilon in epsilons:
        print(f"Running FGSM attack with epsilon={epsilon}.")
        correct = 0
        total = 0
        all_adv_examples = []

        for images, labels in tqdm(test_loader):
            images, labels = images.to(next(encoder.parameters()).device), labels.to(
                next(encoder.parameters()).device
            )
            one_hot_labels = torch.nn.functional.one_hot(
                labels, num_classes=dimY
            ).float()

            # Generate adversarial examples
            adv_images = fast_gradient_method(
                enc_conv, images, eps=epsilon, norm=np.inf
            )
            all_adv_examples.append(adv_images)

            # Evaluate accuracy using Bayes classifier
            y_pred = bayes_classifier(
                adv_images, enc, dec, ll, dimY, lowerbound=lowerbound, K=10, beta=1.0
            )
            correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        accuracies.append(accuracy)

        # Save some adversarial examples for visualization
        adv_examples_to_plot = torch.cat(all_adv_examples)[
            :100
        ]  # Take first 100 examples
        print(adv_examples_to_plot.shape)
        plot_images(
            adv_examples_to_plot.detach().cpu(),
            shape=(28, 28),
            path=save_dir,
            filename=f"{data_name}_{vae_type}_checkpoint_{checkpoint_index}_fgsm_eps_{epsilon}",
            n_rows=10,
            color=False,
        )

    # Plot accuracy vs epsilon
    plt.figure()
    plt.plot(epsilons, accuracies, marker="o")
    plt.title(f"Accuracy vs. Epsilon ({vae_type}, Checkpoint {checkpoint_index})")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(
        os.path.join(
            save_dir,
            f"{data_name}_{vae_type}_checkpoint_{checkpoint_index}_accuracy_vs_epsilon.png",
        )
    )
    plt.close()
    print("Saved accuracy plot and adversarial examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform FGSM attack on a trained VAE."
    )
    parser.add_argument(
        "--vae_type", type=str, required=True, help="VAE model type (A, B, ..., G)."
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Checkpoint index for loading the model.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="List of epsilon values for FGSM attack.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results/", help="Directory to save results."
    )

    args = parser.parse_args()
    perform_fgsm_attack(
        data_name="mnist",
        vae_type=args.vae_type,
        checkpoint_index=args.checkpoint,
        epsilons=args.epsilons,
        save_dir=args.save_dir,
    )
