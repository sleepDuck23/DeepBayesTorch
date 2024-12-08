import torch
from cleverhans.torch.attacks import fast_gradient_method, projected_gradient_descent  # PyTorch attacks

def config_fgsm(targeted, adv_ys, eps):
    """
    Configure parameters for FGSM attack.
    """
    params = {
        'eps': eps,
        'clip_min': 0.0,
        'clip_max': 1.0,
        'targeted': targeted,
        'y': adv_ys if targeted else None
    }
    return params


def config_mim(targeted, adv_ys, eps):
    """
    Configure parameters for Momentum Iterative Method (MIM) attack.
    """
    params = {
        'eps': eps,
        'eps_iter': 0.01,
        'nb_iter': 100,
        'decay_factor': 1.0,
        'clip_min': 0.0,
        'clip_max': 1.0,
        'targeted': targeted,
        'y': adv_ys if targeted else None
    }
    return params


def config_cw(targeted, adv_ys, eps):
    """
    Configure parameters for CW attack.
    """
    from art.attacks.evasion import CarliniL2Method
    params = {
        'max_iter': 1000,
        'learning_rate': 0.01,
        'confidence': 0.0,
        'initial_const': eps,
        'targeted': targeted
    }
    return params


def config_pgd(targeted, adv_ys, eps):
    """
    Configure parameters for PGD (Madry) attack.
    """
    params = {
        'eps': eps,
        'eps_iter': 0.01,
        'nb_iter': 40,
        'clip_min': 0.0,
        'clip_max': 1.0,
        'targeted': targeted,
        'y': adv_ys if targeted else None
    }
    return params


def load_attack(model, attack_method, targeted, adv_ys, eps):
    """
    Load and configure an attack method.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adv_ys = adv_ys.to(device) if adv_ys is not None else None

    if attack_method == 'fgsm':
        attack_func = fast_gradient_method
        attack_params = config_fgsm(targeted, adv_ys, eps)
        
    elif attack_method == 'pgd':
        attack_func = projected_gradient_descent
        attack_params = config_pgd(targeted, adv_ys, eps)

    elif attack_method == 'mim':
        from cleverhans.torch.attacks import momentum_iterative_method
        attack_func = momentum_iterative_method
        attack_params = config_mim(targeted, adv_ys, eps)

    elif attack_method == 'cw':
        from art.attacks.evasion import CarliniL2Method
        attack = CarliniL2Method(estimator=model, **config_cw(targeted, adv_ys, eps))
        return attack, None  # CW attack uses its own interface
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")

    return attack_func, attack_params