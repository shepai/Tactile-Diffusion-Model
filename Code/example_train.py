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
X=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/X_Data.npy")[:,:7,:]#normal stroke data
y=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/y_Data.npy", allow_pickle=True).item().toarray()
Xn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/Xn_Data.npy")[:,:7,:]#non linear stroke data
yn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/yn_Data.npy", allow_pickle=True).item().toarray()
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Xn shape:", Xn.shape)
print("yn shape:", yn.shape)
#preprocess it all
mask = np.any(X != 0, axis=(1, 2, 3))
X, y = X[mask], y[mask]

mask = np.any(X != 1, axis=(1, 2, 3))
X, y = X[mask], y[mask]

mask = np.any(Xn != 0, axis=(1, 2, 3))
Xn, yn = Xn[mask], yn[mask]

mask = np.any(Xn != 1, axis=(1, 2, 3))
Xn, yn = Xn[mask], yn[mask]
#resize and reshape the data

def resize(X_,SF=0.3):
    h=int(X_.shape[2]*SF)
    w=int(X_.shape[3]*SF)
    new_X=np.zeros((len(X_),len(X_[0]),h,w))
    for i in range(len(X_)):
        for j in range(X_.shape[1]):
            resized_image = cv2.resize(X_[i][j], (w,h), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            new_X[i][j]=resized_image
    return new_X 
X=resize(X)
X=X.reshape((len(X),X.shape[1]*X.shape[2],X.shape[3]))
Xn=resize(Xn)
Xn=Xn.reshape((len(Xn),Xn.shape[1]*Xn.shape[2],Xn.shape[3]))
print("X shape:",X.shape,"X non linear shape:",Xn.shape)
#train model
dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),torch.tensor(y, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = DenoisingNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")
train(model, optimizer, dataloader, num_steps=20) #train model 
print("Training completed.")
torch.save(model.state_dict(), "/its/home/drs25/Tactile-Diffusion-Model/Data/models/denoising_nn_20.pth")
#look at examples
generate_images(model)