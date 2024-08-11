"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 


import sys

import matplotlib.pyplot as plt
import torch

from neuralop import LpLoss, H1Loss
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models import TFNO
from neuralop.utils import count_model_params

# device = 'cuda'

# %%
train_loss_str, n_epochs, device, n_layers = 'l2loss', 10, 'cuda', 3
import random

random.seed(0)

# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000, batch_size=32,
    test_resolutions=[16, 32], n_tests=[100, 50],
    test_batch_sizes=[32, 32],
    positional_encoding=True
)
data_processor = data_processor.to(device)

# %%
# We create a tensorized FNO model

model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42, n_layers=n_layers)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()



# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=8e-3,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=round(n_epochs * 1.2))

# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

# train_loss = l2loss  # используем при обучении
match train_loss_str:
    case "l2loss":
        train_loss = l2loss
    case "h1loss":
        train_loss = h1loss
    # If an exact match is not confirmed, this last case will be used if provided
    case _:
        train_loss = l2loss

eval_losses = {'h1': h1loss, 'l2': l2loss}

# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# %%
# Create the trainer
trainer = Trainer(model=model, n_epochs=n_epochs,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=1,  # Здесь задаём как часто печатаются логи на экран.
                  use_distributed=False,
                  verbose=True)

# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,  # Передали для использования
              eval_losses=eval_losses)

# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0].cpu(), cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.cpu().squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.squeeze().detach().cpu().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
