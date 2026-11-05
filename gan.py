"""GAN educativa: arquitectura, pérdidas y entrenamiento real sobre Quick, Draw!."""
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import numpy as np
import torch
from torch import nn

CATEGORY = 'smiley face'

# AQUÍ SE CREA LA RED NEURONAL GENERADORA.
# Un vector z de ruido pasa de latent valores a 128, 256 y 784 píxeles.
# Los pesos y sesgos son parámetros aprendibles; latent es un hiperparámetro.
class Generator(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 784), nn.Tanh())
    def forward(self, z):
        # Tanh devuelve intensidades en [-1, 1], como los datos normalizados.
        return self.net(z)

# AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA.
# Recibe 784 píxeles y devuelve un logit: sigmoid(logit) es su puntuación de real.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x)

def load_data(limit):
    # Descargar sólo las primeras filas del .npy público, sin cargar el archivo entero.
    folder = Path(__file__).parent / 'data'
    folder.mkdir(exist_ok=True)
    path = folder / f'smiley_face_{limit}.npy'
    if path.exists():
        return np.load(path, allow_pickle=False)
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + quote(CATEGORY) + '.npy'
    with urlopen(url, timeout=90) as response:
        version = np.lib.format.read_magic(response)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(response)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(response)
        else:
            raise ValueError('Versión NPY no soportada')
        if fortran or shape[1:] != (784,) or dtype != np.dtype('uint8'):
            raise ValueError('Formato de datos inesperado')
        count = min(limit, shape[0])
        raw = bytearray()
        while len(raw) < count * 784:
            chunk = response.read(min(1048576, count * 784 - len(raw)))
            if not chunk:
                raise ValueError('Descarga incompleta; vuelve a intentar')
            raw.extend(chunk)
    data = np.frombuffer(raw, dtype=np.uint8).reshape(count, 784).copy()
    np.save(path, data)
    return data

def train(data, epochs, batch, latent, lr_g, lr_d, seed, report):
    # Semilla y CPU facilitan reproducir el experimento en un equipo sin GPU.
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    g, d = Generator(latent), Discriminator()
    # Adam actualiza pesos y sesgos; cada red tiene su propio optimizador.
    opt_g = torch.optim.Adam(g.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=lr_d, betas=(0.5, 0.999))
    # BCEWithLogitsLoss combina sigmoid y entropía cruzada de forma estable.
    bce = nn.BCEWithLogitsLoss()
    x = torch.tensor(data.astype(np.float32) / 127.5 - 1)
    fixed_noise = torch.randn(16, latent)
    history = []
    for epoch in range(1, epochs + 1):
        sums = np.zeros(4)
        seen = 0
        for ids in torch.randperm(len(x)).split(batch):
            real = x[ids]
            n = len(real)
            # PASO D: reales -> 1; generados -> 0. detach corta el gradiente a G.
            fake = g(torch.randn(n, latent))
            real_logits, fake_logits = d(real), d(fake.detach())
            loss_d = bce(real_logits, torch.ones_like(real_logits)) + bce(fake_logits, torch.zeros_like(fake_logits))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            # PASO G: busca que D asigne 1 a sus dibujos (pérdida no saturante).
            # Congelar pesos de D permite propagar gradientes hacia G sin actualizar D.
            for p in d.parameters():
                p.requires_grad_(False)
            generated_logits = d(g(torch.randn(n, latent)))
            loss_g = bce(generated_logits, torch.ones_like(generated_logits))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            for p in d.parameters():
                p.requires_grad_(True)
            sums += n * np.array([loss_g.item(), loss_d.item(), real_logits.sigmoid().mean().item(), fake_logits.sigmoid().mean().item()])
            seen += n
        row = dict(zip(['G', 'D', 'D(real)', 'D(G(z))'], (sums / seen).tolist()))
        row['época'] = epoch
        history.append(row)
        with torch.no_grad():
            samples = ((g(fixed_noise).reshape(16, 28, 28) + 1) / 2).numpy()
        report(history, samples)
    return g, d, history
