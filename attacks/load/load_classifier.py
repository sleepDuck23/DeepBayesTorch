import torch

def load_classifier(model_name, data_name, device, path=None, attack_snapshot=False):
    """
    Load a classifier model in PyTorch.
    
    :param model_name: Name of the model (e.g., 'bayes_K10_A', 'fea_K10_F_mid').
    :param data_name: Dataset name (e.g., 'mnist', 'cifar10').
    :param device: Device to load the model onto (e.g., 'cpu', 'cuda').
    :param path: Path to the pre-trained model checkpoint (if applicable).
    :param attack_snapshot: If True, load a model snapshot specific for attacks.
    :return: Loaded PyTorch model.
    """
    if 'bayes' in model_name:
        from load_bayes_classifier import BayesModel  # Assuming a PyTorch implementation exists

        # Parse model parameters
        conv = True
        checkpoint = 0
        vae_type = model_name[-1]
        use_mean = False
        fix_samples = False
        K = int(model_name.split('_')[1][1:])
        
        if conv:
            model_name += '_cnn'
        else:
            model_name += '_mlp'
        
        # Determine latent space dimensionality
        if 'Z' in model_name:
            dimZ = int(model_name.split('_')[2][1:])
        else:
            dimZ = 64 if data_name == 'mnist' else 128
        
        # Instantiate the BayesModel
        model = BayesModel(
            data_name=data_name,
            vae_type=vae_type,
            conv=conv,
            K=K,
            checkpoint=checkpoint,
            attack_snapshot=attack_snapshot,
            use_mean=use_mean,
            fix_samples=fix_samples,
            dimZ=dimZ
        )
    
    elif 'fea' in model_name:
        from load_bayes_classifier_on_fea import BayesModel  # Assuming a PyTorch implementation exists
        
        # Parse model parameters
        conv = True
        checkpoint = 0
        _, K, vae_type, fea_layer = model_name.split('_')
        K = int(K[1:])
        use_mean = False
        fix_samples = False
        
        if conv:
            model_name += '_cnn'
        else:
            model_name += '_mlp'
        
        # Determine latent space dimensionality
        if 'Z' in model_name:
            dimZ = int(model_name.split('_')[2][1:])
        else:
            dimZ = 64 if data_name == 'mnist' else 128
        
        # Instantiate the BayesModel
        model = BayesModel(
            data_name=data_name,
            vae_type=vae_type,
            fea_layer=fea_layer,
            conv=conv,
            K=K,
            checkpoint=checkpoint,
            attack_snapshot=attack_snapshot,
            use_mean=use_mean,
            fix_samples=fix_samples,
            dimZ=dimZ
        )
    
    else:
        raise ValueError('Classifier type not recognized')

    # Move the model to the specified device
    model = model.to(device)

    # Optionally load pre-trained weights
    if path is not None:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
    
    return model