import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import json
import os
import numpy as np
import math
from scipy.spatial import distance
import warnings
warnings.simplefilter('ignore')


class Generator(nn.Module):
    def __init__(self, input_size, latent_dim, layers):
        super(Generator, self).__init__()
        self.layers = layers
        hidden_size = 256

        self.model_2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
        )

        self.model_3 = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size * 2, input_size),
            nn.Tanh(),
        )

        self.model_4 = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size * 4, input_size),
            nn.Tanh(),
        )

    def forward(self, z):
        if self.layers == 2:
            return self.model_2(z)
        elif self.layers == 3:
            return self.model_3(z)
        else:
            return self.model_4(z)


class Discriminator(nn.Module):
    def __init__(self, input_size, layers):
        super(Discriminator, self).__init__()
        self.layers = layers
        hidden_size = 1024

        self.model_2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self.model_3 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.model_4 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.layers == 2:
            return self.model_2(x)
        elif self.layers == 3:
            return self.model_3(x)
        else:
            return self.model_4(x)


def train_generator(discriminator, generator, criterion, data_size):
    real_label = torch.ones(data_size, 1).to(device)

    fake_data = torch.randn(data_size, latent_size).to(device)
    fake_images = generator(fake_data)

    g_loss = criterion(discriminator(fake_images), real_label)

    # Backprop and optimize
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


def train_discriminator(discriminator, generator, criterion, data, data_size):
    real_label = torch.ones(data_size, 1).to(device)
    fake_label = torch.zeros(data_size, 1).to(device)

    # For real images
    real_images = discriminator(data)
    real_loss = criterion(real_images, real_label)

    # For fake images
    z = torch.randn(data_size, latent_size).to(device)
    fake_data = generator(z)
    fake_images = discriminator(fake_data)
    fake_loss = criterion(fake_images, fake_label)

    # Combine losses
    d_loss = real_loss + fake_loss

    # Backpropagation and optimize
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_images, fake_images


def test_generator(generator, epoch):
    with torch.no_grad():
        fake_data = torch.randn(BATCH_SIZE, latent_size).to(device)
        fake_images = generator(fake_data)
        fake_images = fake_images.view(BATCH_SIZE, 1, 28, 28)[:64]
        generated_img = make_grid(fake_images, padding=2, normalize=True)
        file_path = OUT_FOLDER + '/fake_images_epoch{0:0=4d}.png'.format(epoch + 1)
        save_image(generated_img, file_path)


def jsd_loss(true_data, fake_data):
    loss = distance.jensenshannon(true_data, fake_data, base=2.0)
    loss = [x for x in loss if not math.isnan(x) and not math.isinf(x)]
    return np.mean(loss)


def run(generator, discriminator):
    losses = {'gen_losses': [],
              'dis_losses': [],
              'jsd_losses': []
              }

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1} of {N_EPOCHS}")

        criterion = nn.BCELoss()
        generator.train()
        discriminator.train()

        d_losses, g_losses, jsd_losses = [], [], []

        for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
            data_size = data.size(0)
            data = data.view(data_size, -1).to(device)

            d_loss, real_score, fake_score = train_discriminator(discriminator, generator, criterion, data, data_size)
            d_losses.append(d_loss.data.item())

            g_loss, fake_images = train_generator(discriminator, generator, criterion, data_size)
            g_losses.append(g_loss.data.item())

            loss = jsd_loss(data.cpu().detach(), fake_images.cpu().detach())
            jsd_losses.append(loss)

        losses['gen_losses'].append(np.mean(g_losses))
        losses['dis_losses'].append(np.mean(d_losses))
        losses['jsd_losses'].append(np.float64(np.mean(jsd_losses)))
        print(f'g_loss: {np.mean(g_losses)}, d_loss: {np.mean(d_losses)}, jds_loss: {np.mean(jsd_losses)}')

        test_generator(generator, epoch)

    return losses


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    BATCH_SIZE = 64
    DROPOUT = 0.3
    LR = 2e-4
    N_EPOCHS = 100
    MNIST_SIZE = 28 * 28

    latent_size_list = [2, 64]
    layers_list = [2, 3, 4]

    params = list(itertools.product(latent_size_list, layers_list))

    for i, param in enumerate(params):
        torch.manual_seed(42)
        OUT_FOLDER = './output/gan_mnist_' + str(param).replace('(', '').replace(')', '').replace(', ', '_')
        os.makedirs(OUT_FOLDER, exist_ok=True)

        latent_size, n_layers = param
        print(f'Test {i + 1}/{len(params)} - Latent size: {latent_size}, # of layers: {n_layers}')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        gen_model = Generator(MNIST_SIZE, latent_size, n_layers)
        dis_model = Discriminator(MNIST_SIZE, n_layers)
        g_optimizer = optim.Adam(gen_model.parameters(), lr=LR)
        d_optimizer = optim.Adam(dis_model.parameters(), lr=LR)

        gen_model.to(device)
        dis_model.to(device)

        results = run(gen_model, dis_model)
        json_object = json.dumps(results, indent=4)
        filename = str(param).replace('(', '').replace(')', '').replace(', ', '_')
        path = OUT_FOLDER + '/results.json'
        with open(path, "w") as outfile:
            outfile.write(json_object)
