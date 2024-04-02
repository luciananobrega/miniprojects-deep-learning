# Ref for vae_loss: https://pyimagesearch.com/2023/10/02/a-deep-dive-into-variational-autoencoders-with-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np
import json
import os


class VAE(nn.Module):
    def __init__(self, input_size, latent_dim, layers):
        super(VAE, self).__init__()
        self.layers = layers
        hidden_size = 512
        if n_layers == 2:
            # for encoder
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, latent_dim * 2)  # Two outputs for mean and log variance
            # for decoder
            self.fc3 = nn.Linear(latent_dim, hidden_size)
            self.fc4 = nn.Linear(hidden_size, input_size)
        elif n_layers == 3:
            # for encoder
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
            self.fc3 = nn.Linear(hidden_size * 2, latent_dim * 2)  # Two outputs for mean and log variance
            # for decoder
            self.fc4 = nn.Linear(latent_dim, hidden_size * 2)
            self.fc5 = nn.Linear(hidden_size * 2, hidden_size)
            self.fc6 = nn.Linear(hidden_size, input_size)
        else:
            # for encoder
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
            self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
            self.fc4 = nn.Linear(hidden_size * 4, latent_dim * 2)  # Two outputs for mean and log variance
            # for decoder
            self.fc5 = nn.Linear(latent_dim, hidden_size * 4)
            self.fc6 = nn.Linear(hidden_size * 4, hidden_size * 2)
            self.fc7 = nn.Linear(hidden_size * 2, hidden_size)
            self.fc8 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encoder
        if self.layers == 2:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.layers == 3:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)

        mu, log_var = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        # Decoder
        if self.layers == 2:
            z = F.relu(self.fc3(z))
            z = self.fc4(z)
        elif self.layers == 3:
            z = F.relu(self.fc4(z))
            z = F.relu(self.fc5(z))
            z = self.fc6(z)
        else:
            z = F.relu(self.fc5(z))
            z = F.relu(self.fc6(z))
            z = F.relu(self.fc7(z))
            z = self.fc8(z)

        reconstruction = torch.sigmoid(z).to(device)

        return reconstruction, mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std


def vae_loss(reconstruction, x, mu, log_var):
    # Reconstruction loss (binary cross-entropy)
    bce_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')

    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss
    total_loss = bce_loss + kl_divergence

    return bce_loss, kl_divergence, total_loss


def train_vae(model):
    model.train()

    running_loss = []
    running_loss_bce = []
    running_loss_kl = []

    print('Training')
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.view(-1, MNIST_SIZE).to(device)  # Flatten MNIST images
        optimizer.zero_grad()

        reconstruction, mu, log_var = model(data)
        bce_loss, kl_loss, loss = vae_loss(reconstruction, data, mu, log_var)
        running_loss.append(loss.item())
        running_loss_bce.append(bce_loss.item())
        running_loss_kl.append(kl_loss.item())

        loss.backward()
        optimizer.step()

    train_loss = np.mean(running_loss)
    bce_loss = np.mean(running_loss_bce)
    kl_loss = np.mean(running_loss_kl)

    return bce_loss, kl_loss, train_loss


def evaluate_vae(model, epoch):
    model.eval()
    running_loss = []
    running_loss_bce = []
    running_loss_kl = []

    print('Evaluating')
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
            data = data.view(-1, MNIST_SIZE).to(device)  # Flatten MNIST images
            reconstruction, mu, log_var = model(data)
            bce_loss, kl_loss, loss = vae_loss(reconstruction, data, mu, log_var)
            running_loss.append(loss.item())
            running_loss_bce.append(bce_loss.item())
            running_loss_kl.append(kl_loss.item())

            if batch_idx == int(len(test_loader.dataset) / BATCH_SIZE) - 1:
                recon_batch_ = reconstruction.view(BATCH_SIZE, 1, 28, 28)[:64]
                generated_img = make_grid(recon_batch_, padding=2, normalize=True)
                file_path = OUT_FOLDER + '/fake_images_epoch{0:0=4d}.png'.format(epoch + 1)
                save_image(generated_img, file_path)

    train_loss = np.mean(running_loss)
    bce_loss = np.mean(running_loss_bce)
    kl_loss = np.mean(running_loss_kl)

    return bce_loss, kl_loss, train_loss


def run(model):
    losses = {'train_losses': [],
              'train_losses_bce': [],
              'train_losses_kl': [],
              'test_losses': [],
              'test_losses_bce': [],
              'test_losses_kl': []
              }

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1} of {N_EPOCHS}")
        train_epoch_loss_BCE, train_epoch_loss_KLD, train_epoch_loss = train_vae(model)
        losses['train_losses'].append(train_epoch_loss)
        losses['train_losses_bce'].append(train_epoch_loss_BCE)
        losses['train_losses_kl'].append(train_epoch_loss_KLD)

        test_epoch_loss_BCE, test_epoch_loss_KLD, test_epoch_loss = evaluate_vae(model, epoch)
        losses['test_losses'].append(test_epoch_loss)
        losses['test_losses_bce'].append(test_epoch_loss_BCE)
        losses['test_losses_kl'].append(test_epoch_loss_KLD)

    return losses


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    BATCH_SIZE = 64
    LR = 1e-3
    N_EPOCHS = 100
    MNIST_SIZE = 28 * 28

    latent_size_list = [2, 64]
    layers_list = [2, 3, 4]

    params = list(itertools.product(latent_size_list, layers_list))

    for i, param in enumerate(params):
        torch.manual_seed(42)
        OUT_FOLDER = './output/vae_mnist_' + str(param).replace('(', '').replace(')', '').replace(', ', '_')
        os.makedirs(OUT_FOLDER, exist_ok=True)

        latent_size, n_layers = param
        print(f'Test {i + 1}/{len(params)} - Latent size: {latent_size}, # of layers: {n_layers}')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        vae_model = VAE(MNIST_SIZE, latent_size, n_layers)
        optimizer = optim.Adam(vae_model.parameters(), lr=LR)

        vae_model.to(device)

        results = run(vae_model)
        json_object = json.dumps(results, indent=4)
        path = OUT_FOLDER + '/results.json'
        with open(path, "w") as outfile:
            outfile.write(json_object)
