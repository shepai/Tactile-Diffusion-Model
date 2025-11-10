from DiffusionPipeline import DenoisingNN, generate_images, train
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = DenoisingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
train(model, optimizer, dataloader, num_steps=10)
print("Training completed.")

generate_images(model)