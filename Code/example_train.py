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
#load in data and process
X=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/X_data.npy")[:,:7,:].reshape((200,7*355,328))
y=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/y_data.npy")
Xn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/Xn_data.npy")[:,:7,:].reshape((200,7*355,328))
yn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/yn_data.npy")
mask = np.any(XB != 0, axis=(1, 2, 3))
XB = XB[mask]
yB = yB[mask]
print(XB.shape)
mask = np.any(XB != 1, axis=(1, 2, 3))
XB = XB[mask]
yB = yB[mask]

#train model
dataset = torch.utils.data.TensorDataset(torch.tensor(X))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = DenoisingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
train(model, optimizer, dataloader, num_steps=100)
print("Training completed.")

#look at examples
generate_images(model)