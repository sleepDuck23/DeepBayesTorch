from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import sys, os
import pickle
PATH = '../'
sys.path.extend([PATH+'alg/', PATH+'models/', PATH+'utils/'])

class BayesModel(nn.Module):
    def __init__(self, data_name, vae_type, conv=True, K=1, checkpoint=0, 
                 attack_snapshot=False, use_mean=False, fix_samples=False, no_z=False,
                 dimZ=None, device="cpu"):
        super(BayesModel, self).__init__()

        self.device = device
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
        elif data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
        else:
            raise ValueError("Unsupported dataset")

        self.num_labels = 10
        self.conv = conv
        self.K = K

        # Handle configuration flags
        if no_z:
            use_mean = attack_snapshot = fix_samples = False
        if fix_samples:
            use_mean = attack_snapshot = no_z = False
        if use_mean:
            attack_snapshot = fix_samples = no_z = False
        if attack_snapshot:
            use_mean = fix_samples = no_z = False

        print('Settings:')
        print(f'no_z: {no_z}, use_mean: {use_mean}, fix_samples: {fix_samples}, attack_snapshot: {attack_snapshot}')

        # Load the Bayesian classifier components
        self.model, self.eval_test_ll, self.enc, self.dec = load_bayes_classifier(
            data_name, vae_type, K, checkpoint, conv, attack_snapshot, use_mean, 
            fix_samples, no_z, dimZ, device
        )
        self.use_mean = use_mean
        self.attack_snapshot = attack_snapshot
        self.fix_samples = fix_samples
        self.no_z = no_z

    def forward(self, X, softmax=False):
        # Flatten input if not convolutional
        if not self.conv:
            X = X.view(X.size(0), -1)

        results = self.model(X)
        if softmax:
            if self.attack_snapshot:
                K = results.size(0)
                if K > 1:
                    results = logsumexp(results) - torch.log(torch.tensor(float(K)))
                else:
                    results = results[0]
            results = torch.softmax(results, dim=-1)
        return results

    def comp_test_ll(self, x, y, K=1):
        # Flatten input if not convolutional
        if not self.conv:
            x = x.view(x.size(0), -1)
        return self.eval_test_ll(x, y, K)

def logsumexp(x):
    """Compute log-sum-exp."""
    x_max = torch.max(x, dim=0)[0]
    x_ = x - x_max
    tmp = torch.log(torch.clamp(torch.sum(torch.exp(x_), dim=0), min=1e-20))
    return tmp + x_max

def load_bayes_classifier(data_name, vae_type, K, checkpoint=0, conv=True, 
                          attack_snapshot=False, use_mean=False, fix_samples=False, no_z=False,
                          dimZ=None, device="cpu"):
    """Load the Bayesian classifier model and associated components."""
    if data_name == 'mnist':
        input_shape = (1, 28, 28)
        dimX = 28**2
        dimY = 10
        dimZ = 64 if dimZ is None else dimZ
        ll = 'l2'
        beta = 1.0
        from conv_generator_mnist import generator
        from conv_encoder_mnist import encoder_gaussian as encoder
    elif data_name == 'cifar10':
        input_shape = (3, 32, 32)
        dimX = 32**2 * 3
        dimY = 10
        dimZ = 128 if dimZ is None else dimZ
        ll = 'l1'
        beta = 1.0
        from conv_generator_cifar10 import generator
        from conv_encoder_cifar10 import encoder_gaussian as encoder
    else:
        raise ValueError("Unsupported dataset")

    # Load generator and encoder
    dec = generator(input_shape, dimZ, dimY, device)
    enc_conv, enc_mlp = encoder(input_shape, dimZ, dimY, device)

    # Load parameters if checkpoint is provided
    if checkpoint > 0:
        load_params(dec, enc_conv, enc_mlp, checkpoint, data_name, vae_type, dimZ)

    def classifier(x):
        return bayes_classifier(x, [enc_conv, enc_mlp], dec, ll, dimY, dimZ, beta, K)

    def eval_test_ll(x, y, K):
        fea = enc_conv(x)
        bound = lowerbound(x, fea, y, enc_mlp, dec, ll, K, beta, use_mean)
        return torch.mean(bound)

    return classifier, eval_test_ll, [enc_conv, enc_mlp], dec

def bayes_classifier(x, enc, dec, ll, dimY, dimZ, beta, K):
    """Define the Bayesian classifier."""
    enc_conv, enc_mlp = enc
    fea = enc_conv(x)
    logpxy = []
    for i in range(dimY):
        y = torch.zeros((x.size(0), dimY), device=x.device)
        y[:, i] = 1
        bound = lowerbound(x, fea, y, enc_mlp, dec, ll, K, beta)
        logpxy.append(bound.unsqueeze(1))
    logpxy = torch.cat(logpxy, dim=1)
    return logpxy

def load_params(dec, enc_conv, enc_mlp, checkpoint, data_name, vae_type, dimZ):
    """Load pre-trained parameters."""
    path_name = f'../save/{data_name}_conv_vae_{vae_type}_{dimZ}/'
    filename = f"{path_name}checkpoint_{checkpoint}.pkl"
    assert os.path.isfile(filename), f"Checkpoint not found: {filename}"
    with open(filename, 'rb') as f:
        param_dict = pickle.load(f)

    # Load parameters into the decoder and encoders
    dec.load_state_dict(param_dict["dec"])
    enc_conv.load_state_dict(param_dict["enc_conv"])
    enc_mlp.load_state_dict(param_dict["enc_mlp"])
    print(f"Parameters loaded from {filename}")