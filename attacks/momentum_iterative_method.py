"""The MomentumIterativeMethod attack."""

import numpy as np
import torch
from cleverhans.torch.utils import optimize_linear


def momentum_iterative_method(
    model_fn,
    x,
    eps=0.3,
    eps_iter=0.06,
    nb_iter=10,
    norm=np.inf,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    decay_factor=1.0,
    sanity_checks=True,
):
    """
    PyTorch implementation of Momentum Iterative Method (MIM).
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: maximum distortion of adversarial example compared to original input.
    :param eps_iter: step size for each attack iteration.
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum input component value.
    :param clip_max: Maximum input component value.
    :param y: Tensor with true labels. If targeted is true, provide the target label. Otherwise,
              use true labels or model predictions as ground truth.
    :param targeted: bool. If True, create a targeted attack; otherwise untargeted.
    :param decay_factor: Decay factor for the momentum term.
    :param sanity_checks: bool. If True, perform sanity checks on inputs and outputs.
    :return: Adversarial examples as a PyTorch tensor.
    """

    if norm == 1:
        raise NotImplementedError("This attack hasn't been tested for norm=1.")

    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be np.inf, 1, or 2.")

    # Ensure input is a leaf tensor
    x = x.clone().detach().requires_grad_(True)

    if y is None:
        # Use model predictions as ground truth
        with torch.no_grad():
            _, y = torch.max(model_fn(x), 1)

    # Initialize variables
    momentum = torch.zeros_like(x)
    adv_x = x.clone()

    for i in range(nb_iter):
        # Compute loss
        logits = model_fn(adv_x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        if targeted:
            loss = -loss

        # Compute gradient
        grad = torch.autograd.grad(loss, adv_x, retain_graph=False, create_graph=False)[
            0
        ]

        # Normalize gradient
        red_ind = list(range(1, len(grad.shape)))  # Reduce indices except batch
        grad = grad / torch.maximum(
            torch.tensor(1e-12, device=grad.device, dtype=grad.dtype),
            torch.mean(torch.abs(grad), dim=red_ind, keepdim=True),
        )

        # Update momentum
        momentum = decay_factor * momentum + grad

        # Compute perturbation and update adversarial example
        optimal_perturbation = optimize_linear(momentum, eps_iter, norm)
        adv_x = adv_x + optimal_perturbation

        # Project perturbation to epsilon ball and clip
        eta = adv_x - x
        eta = torch.clamp(eta, -eps, eps) if norm == np.inf else eta
        adv_x = x + eta

        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)

    # Perform sanity checks if enabled
    if sanity_checks:
        if clip_min is not None:
            assert torch.all(adv_x >= clip_min)
        if clip_max is not None:
            assert torch.all(adv_x <= clip_max)

    return adv_x
