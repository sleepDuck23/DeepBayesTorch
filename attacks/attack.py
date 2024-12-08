from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cleverhans.torch.attacks import projected_gradient_descent, fast_gradient_method  # PyTorch attacks in CleverHans
import pickle

# Custom imports for loading models and data
from load_classifier import load_classifier  # Replace this with a PyTorch version
from model_eval import model_eval  # Replace this with a PyTorch evaluation version


def test_attacks(data_name, model_name, attack_method, eps, batch_size=100, targeted=False, save=False):
    """
    Test adversarial attacks on a PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(1234)

    # Load data
    if data_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    elif data_name == 'cifar10' or data_name == 'plane_frog':
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        if data_name == 'plane_frog':
            test_dataset = [(x, y) for x, y in test_dataset if y in [0, 6]]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = load_classifier(model_name, data_name)
    model = model.to(device)
    model.eval()

    print(f"Loaded model {model_name} for dataset {data_name}.")

    # Initialize attack
    if attack_method == 'pgd':
        attack_func = projected_gradient_descent
    elif attack_method == 'fgsm':
        attack_func = fast_gradient_method
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")

    print(f"Using attack: {attack_method} with eps = {eps}")

    adv_examples = []
    true_labels = []
    adv_labels = []

    # Perform attack
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        if targeted:
            # Craft targeted adversarial examples
            target_labels = torch.randint(0, 10, target.shape).to(device)  # Random target labels
            adv_data = attack_func(model, data, eps, targeted=targeted, y=target_labels)
            adv_labels.append(target_labels.cpu().numpy())
        else:
            # Craft untargeted adversarial examples
            adv_data = attack_func(model, data, eps)

        adv_examples.append(adv_data.cpu().detach().numpy())
        true_labels.append(target.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    adv_examples = np.concatenate(adv_examples, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    if targeted:
        adv_labels = np.concatenate(adv_labels, axis=0)

    # Evaluate adversarial examples
    accuracy = model_eval(model, test_loader, device)
    print(f"Clean test accuracy: {accuracy:.4f}")

    adv_loader = DataLoader(list(zip(adv_examples, true_labels)), batch_size=batch_size, shuffle=False)
    adv_accuracy = model_eval(model, adv_loader, device)
    success_rate = (1 - adv_accuracy) * 100 if not targeted else adv_accuracy * 100
    print(f"Adversarial attack success rate: {success_rate:.4f}%")

    # Save results
    if save:
        results_dir = "raw_attack_results"
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{data_name}_{attack_method}_eps{eps:.2f}{'_targeted' if targeted else '_untargeted'}.pkl"
        with open(os.path.join(results_dir, filename), 'wb') as f:
            pickle.dump({
                'adv_examples': adv_examples,
                'true_labels': true_labels,
                'adv_labels': adv_labels if targeted else None
            }, f)
        print(f"Results saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PyTorch adversarial attack experiments.')
    parser.add_argument('--batch_size', '-B', type=int, default=100)
    parser.add_argument('--data', '-D', type=str, default='plane_frog')
    parser.add_argument('--targeted', '-T', action='store_true', default=False)
    parser.add_argument('--attack', '-A', type=str, default='pgd')
    parser.add_argument('--eps', '-e', type=float, default=0.1)
    parser.add_argument('--victim', '-V', type=str, default='bnn_K10')
    parser.add_argument('--save', '-S', action='store_true', default=False)

    args = parser.parse_args()
    test_attacks(data_name=args.data,
                 model_name=args.victim,
                 attack_method=args.attack,
                 eps=args.eps,
                 batch_size=args.batch_size,
                 targeted=args.targeted,
                 save=args.save)