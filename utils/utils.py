import os

import torch


def load_data(data_name, path, labels=None, conv=False, seed=0):
    """
    Loads dataset based on its name.
    Args:
        data_name (str): Name of the dataset ('mnist', 'omni', 'cifar10').
        path (str): Path to the dataset directory.
        labels (list, optional): List of labels to filter.
        conv (bool): Whether to retain 4D shape for convolutional models.
        seed (int): Random seed for reproducibility.
    Returns:
        tuple: data_train, data_test, labels_train, labels_test
    """
    if data_name == "mnist":
        from .mnist import CustomMNISTDataset

        train_dataset = CustomMNISTDataset(
            path=path, train=True, digits=labels, conv=conv
        )
        test_dataset = CustomMNISTDataset(
            path=path, train=False, digits=labels, conv=conv
        )

    elif data_name == "cifar10":
        from .cifar10 import CustomCIFAR10Dataset

        train_dataset = CustomCIFAR10Dataset(
            path=path, train=True, labels=labels, conv=conv, seed=seed
        )
        test_dataset = CustomCIFAR10Dataset(
            path=path, train=False, labels=labels, conv=conv, seed=seed
        )
    elif data_name == "gtsrb":
        from .gtsrb import GermanTrafficSignDataset

        train_dataset = GermanTrafficSignDataset(
            root_dir=path, train=True, labels=labels
        )
        test_dataset = GermanTrafficSignDataset(
            root_dir=path, train=False, labels=labels
        )
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")

    return train_dataset, test_dataset


def save_params(model, filename, checkpoint):
    """
    Save model parameters to a file.
    Args:
        model (torch.nn.Module): PyTorch model.
        filename (str): Path to save the parameters.
        checkpoint (int): Checkpoint index for versioning.
    """
    encoder, generator = model
    filename = f"{filename}_{checkpoint}.pth"
    state_dict = {"encoder": encoder.state_dict(), "generator": generator.state_dict()}
    torch.save(state_dict, filename)
    print(f"Parameters saved at {filename}")


def load_params(model, filename, checkpoint):
    """
    Load model parameters from a file.
    Args:
        model (torch.nn.Module): PyTorch model.
        filename (str): Path to load the parameters from.
        checkpoint (int): Checkpoint index for versioning.
    """
    filename = f"{filename}_{checkpoint}.pth"
    encoder, generator = model
    if os.path.exists(filename):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(filename, map_location=device)
        encoder.load_state_dict(state_dict["encoder"])
        generator.load_state_dict(state_dict["generator"])
        print(f"Loaded parameters from {filename}")
    else:
        print(f"Checkpoint {filename} not found. Skipping parameter loading.")


def init_variables(model, optimizer=None):
    """
    Initialize model parameters and optionally an optimizer.
    Args:
        model (torch.nn.Module): PyTorch model.
        optimizer (torch.optim.Optimizer, optional): Optimizer to reset.
    """
    model.apply(reset_weights)
    if optimizer:
        optimizer.state = {}


def reset_weights(layer):
    """
    Resets weights of a layer.
    """
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
