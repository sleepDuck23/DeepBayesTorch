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
    filename = f"{filename}_{checkpoint}.pth"
    torch.save(model.state_dict(), filename)
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
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
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


if __name__ == "__main__":
    train_dataset, test_dataset = load_data(
        data_name="cifar10", path="./data", labels=[0, 1], conv=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    from torch import nn, optim

    model = nn.Sequential(
        nn.Linear(32 * 32 * 3, 128), nn.ReLU(), nn.Linear(128, 10)
    )  # Example model
    optimizer = optim.Adam(model.parameters())
    init_variables(model, optimizer)

    # Save model parameters
    save_params(model, filename="my_model", checkpoint=1)

    # Load model parameters
    load_params(model, filename="my_model", checkpoint=1)

    # Iterate through the data
    for images, labels in train_loader:
        print(images.shape)
        break
