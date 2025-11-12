from DiffusionPipeline import DenoisingNN, generate_images, train
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
#load in data and process
X=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/X_Data.npy")[:,:7,:]#.reshape((200,7*355,328))
y=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/y_Data.npy")
Xn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/Xn_Data.npy")[:,:7,:]#.reshape((200,7*355,328))
yn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/yn_Data.npy")
#preprocess it all
mask = np.any(X != 0, axis=(1, 2, 3))
X = X[mask]
y = y[mask]
mask = np.any(X != 1, axis=(1, 2, 3))
X = X[mask]
y = y[mask]
mask = np.any(Xn != 0, axis=(1, 2, 3))
Xn = Xn[mask]
yn = yn[mask]
mask = np.any(Xn != 1, axis=(1, 2, 3))
Xn = Xn[mask]
yn = yn[mask]
print("X shape:",X.shape,"X non linear shape:",Xn.shape)
def resize(X_):
    h=int(X.shape[2]*0.3)
    w=int(X.shape[3]*0.3)
    new_X=np.zeros((len(X),len(X[0]),h,w))
    #resize and reshape the data
    for i in range(len(X)):
        for j in range(len(X.shape[1])):
            resized_image = cv2.resize(X[i][j], (h,w), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            new_X[i][j]=resized_image
    return new_X 
X=resize(X).reshape((len(X),X.shape[1]*X.shape[2],X.shape[3]))
Xn=resize(Xn).reshape((len(Xn),Xn.shape[1]*Xn.shape[2],Xn.shape[3]))
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