# DeepBayesTorch

Pytorch implementation of the article
[Are Generative Classifiers More Robust to Adversarial Attacks?](https://arxiv.org/abs/1802.06552).

## Install

Create a Python3.12.2 virtual environment

```bash
pyenv virtualenv 3.12.2 bayes
pyenv activate byes
```

Install the dependencies using `poetry`

```bash
pip install poetry
poetry install
```

## Usage

### Train a generative classifier

Example of training classifier "A" on MNIST

```bash
python vae_mnist.py A -1
```

Once trained, you can run

```bash
python vae_mnist.py A 0
```

to resume training to saved checkpoint 0.

### Perform attacks on generative classifiers

- [ ] Todo
