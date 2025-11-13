from DiffusionPipeline import DenoisingNN, generate_images, train
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2

#load in the data
Xn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/Xn_Data.npy")[:,:7,:]#non linear stroke data
yn=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/yn_Data.npy", allow_pickle=True).item().toarray()

mask = np.any(Xn != 0, axis=(1, 2, 3))
Xn, yn = Xn[mask], yn[mask]

mask = np.any(Xn != 1, axis=(1, 2, 3))
Xn, yn = Xn[mask], yn[mask]

def resize(X_,SF=0.3):
    h=int(X_.shape[2]*SF)
    w=int(X_.shape[3]*SF)
    new_X=np.zeros((len(X_),len(X_[0]),h,w))
    for i in range(len(X_)):
        for j in range(X_.shape[1]):
            resized_image = cv2.resize(X_[i][j], (w,h), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            new_X[i][j]=resized_image
    return new_X 
Xn=resize(Xn)
Xn=Xn.reshape((len(Xn),Xn.shape[1]*Xn.shape[2],Xn.shape[3]))
Xn=torch.tensor(Xn, dtype=torch.float32)

model = DenoisingNN()
model.load_state_dict(torch.load("/its/home/drs25/Tactile-Diffusion-Model/Data/models/denoising_nn_20.pth"))
model.eval()
#model.to("cuda")

converted=model(Xn.view(Xn.size(0), -1))
converted=converted.cpu().detach().numpy()
np.save("/its/home/drs25/Tactile-Diffusion-Model/Data/converted_Xn_20",Xn)

