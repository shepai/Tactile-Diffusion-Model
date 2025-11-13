import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from DiffusionPipeline import DenoisingNN, generate_images, train
from sklearn.model_selection import train_test_split

################################################
# Convert linear dataset
################################################
X=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/X_Data.npy")[:,:7,:]#normal stroke data
y=np.load("/its/home/drs25/Tactile-Diffusion-Model/Data/y_Data.npy", allow_pickle=True).item().toarray()
mask = np.any(X != 0, axis=(1, 2, 3))
X, y = X[mask], y[mask]
mask = np.any(X != 1, axis=(1, 2, 3))
X, y = X[mask], y[mask]
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
X=torch.tensor(X.reshape((len(X),X.shape[1]*X.shape[2],X.shape[3]))).float()

model = DenoisingNN()
model.load_state_dict(torch.load("/its/home/drs25/Tactile-Diffusion-Model/Data/models/denoising_nn.pth"))
model.eval()
#model.to("cuda")

with torch.no_grad():
    converted = model(X.view(X.size(0), -1))
converted = converted.detach()

##############################################
# Train classifier
##############################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#train model
converted = converted / 255
y_train = y.astype(int)  # Ensure labels are integers

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    converted, y_train, test_size=0.2, random_state=42, shuffle=True
)

# Initialize and train Random Forest classifier
classifier = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=None,          # expand until all leaves are pure
    random_state=42,
    n_jobs=-1                # use all CPU cores
)

print("Classifier data shape:", converted.shape, y.shape)
classifier.fit(X_train, y_train)

# Evaluate model
y_pred = classifier.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc:.4f}")

# Optional: Save model using joblib or pickle
import joblib
joblib.dump(classifier, "/its/home/drs25/Tactile-Diffusion-Model/Data/models/classifier_rf.pkl")

