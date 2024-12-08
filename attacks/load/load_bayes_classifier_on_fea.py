import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys, os
PATH = '../'
sys.path.extend([PATH+'alg/', PATH+'models/', PATH+'utils/'])


class BayesModel:
    def __init__(self, data_name, vae_type, fea_layer, conv=True, K=1, checkpoint=0, 
                 attack_snapshot=False, use_mean=False, fix_samples=False, no_z=False,
                 dimZ=None):
        if data_name == 'mnist':
            self.num_channels = 1
            self.image_size = 28
        if data_name == 'cifar10':
            self.num_channels = 3
            self.image_size = 32
        self.num_labels = 10
        self.conv = conv
        self.K = K
        
        if no_z:
            use_mean = False
            attack_snapshot = False
            fix_samples = False
        if fix_samples:
            use_mean = False
            attack_snapshot = False
            no_z = False
        if use_mean:
            attack_snapshot = False
            fix_samples = False
            no_z = False
        if attack_snapshot:
            use_mean = False
            fix_samples = False
            no_z = False

        print('settings:')
        print('feature layer', fea_layer)
        print('no_z', no_z)
        print('use_mean', use_mean)
        print('fix_samples', fix_samples)
        print('attack_snapshot', attack_snapshot)

        self.model, self.eval_test_ll, self.enc, self.dec, self.fea_op = load_bayes_classifier(
            data_name, vae_type, fea_layer, K, checkpoint, conv=conv, 
            attack_snapshot=attack_snapshot, use_mean=use_mean, 
            fix_samples=fix_samples, no_z=no_z, dimZ=dimZ
        )
        self.use_mean = use_mean
        self.attack_snapshot = attack_snapshot
        self.fix_samples = fix_samples
        self.no_z = no_z

    def predict(self, data, softmax=False):
        if not self.conv:
            data = data.view(data.size(0), -1)
        results = self.model(data)
        if softmax:
            if self.attack_snapshot:
                K = results.size(0)
                if K > 1:
                    results = logsumexp(results) - torch.log(torch.tensor(float(K)))
                else:
                    results = results[0]
            results = F.softmax(results, dim=-1)
        return results

    def comp_test_ll(self, x, y, K=1):
        if not self.conv:
            x = x.view(x.size(0), -1)
        return self.eval_test_ll(x, y, K)


def logsumexp(x):
    x_max = torch.max(x, dim=0, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_max + torch.log(torch.sum(x_exp, dim=0))


class FeatureExtractor(nn.Module):
    def __init__(self, cnn_model, fea_layer):
        super().__init__()
        self.model = cnn_model
        self.fea_layer = fea_layer
        self.layers = nn.Sequential(*list(cnn_model.children())[:fea_layer])

    def forward(self, x):
        x = x * 255.0  # Normalize
        x = self.layers(x)
        return x.view(x.size(0), -1)


def bayes_classifier(x, enc, dec, ll, dimY, dimZ, lowerbound, K=1, beta=1.0, use_mean=False, 
                     fix_samples=False, snapshot=False, no_z=False, softmax=False, N=None):
    if use_mean:
        K = 1
    fea = enc(x)
    if N is None:
        N = x.size(0)
    logpxy = []
    if no_z:
        z_holder = torch.zeros(N, dimZ)
        K = 1
    else:
        z_holder = None

    for i in range(dimY):
        y = torch.zeros(N, dimY)
        y[:, i] = 1
        bound = lowerbound(x, fea, y, enc, dec, ll, K, IS=False, beta=beta, z=z_holder)
        logpxy.append(bound.unsqueeze(1))

    logpxy = torch.cat(logpxy, dim=1)
    if snapshot:
        logpxy = logpxy.view(K, N, dimY)
    elif K > 1:
        logpxy = logpxy.view(K, N, dimY)
        logpxy = logsumexp(logpxy) - torch.log(torch.tensor(float(K)))

    if softmax:
        return F.softmax(logpxy, dim=-1)
    return logpxy


def load_bayes_classifier(data_name, vae_type, fea_layer, K, checkpoint=0, conv=True, 
                          attack_snapshot=False, use_mean=False, fix_samples=False, no_z=False, dimZ=None):
    # Define input dimensions and parameters
    if data_name == 'mnist':
        input_shape = (1, 28, 28)
        dimX = 28 ** 2
        dimY = 10
    elif data_name == 'cifar10':
        input_shape = (3, 32, 32)
        dimX = 32 ** 2 * 3
        dimY = 10

     # then define model
    # note that this is only for cifar10
    if data_name in ['cifar10']:
        if vae_type == 'E':
            from mlp_generator_cifar10_E import generator
        if vae_type == 'F':
            from mlp_generator_cifar10_F import generator
        if vae_type == 'G':
            from mlp_generator_cifar10_G import generator
        from mlp_encoder_cifar10 import encoder_gaussian as encoder

    # Feature extractor
    from vgg_cifar10 import cifar10vgg
    cnn_model = cifar10vgg()  # Placeholder for actual model
    fea_op = FeatureExtractor(cnn_model, fea_layer)

    dimF = fea_op(torch.randn(1, *input_shape)).size(1)
    dimH = 1000
    dimZ = dimZ or 128

    dec = Generator(dimF, dimH, dimZ, dimY)
    enc = Encoder(dimF, dimH, dimZ, dimY)

    # Lowerbound function (define as per your specific setup)
    lowerbound = LowerboundFunction(vae_type)

    # Load model parameters (implement function to load PyTorch state_dict)
    checkpoint_path = f'./path/to/checkpoints/{data_name}_model_{vae_type}_fea_{fea_layer}.pt'
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        fea_op.load_state_dict(state_dict['fea_op'])
        enc.load_state_dict(state_dict['enc'])
        dec.load_state_dict(state_dict['dec'])
        print(f'Loaded weights from {checkpoint_path}')
    else:
        print(f'Checkpoint {checkpoint_path} not found!')

    def classifier(x):
        fea = fea_op(x)
        return bayes_classifier(fea, enc, dec, 'll', dimY, dimZ, lowerbound, K=K, beta=1.0)

    return classifier, None, enc, dec, fea_op


# Additional components (e.g., Generator, Encoder) must be implemented in PyTorch.