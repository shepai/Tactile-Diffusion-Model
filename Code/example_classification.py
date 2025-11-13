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
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(742*98, 742*98//10)
        self.fc2 = nn.Linear(742*98//10, 128)
        self.fc3 = nn.Linear(128, 14)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # No softmax here (CrossEntropyLoss expects logits)
        x = self.fc3(x)  # No softmax here (CrossEntropyLoss expects logits)
        return x

def train_model(model, X_train, y_train, num_epochs=100, lr=0.001):
    """
    Train the model on given training data.
    X_train: torch.Tensor of shape [N, input_size]
    y_train: torch.Tensor of shape [N] (class indices)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch) % 10 == 0:
            predicted = torch.argmax(outputs, axis=1)
            acc = (predicted == torch.argmax(y_train, axis=1)).sum().item() / y_train.size(0)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    print("Training complete.")

def predict(model, X):
    """
    Predict class labels for given input X.
    X: torch.Tensor of shape [N, input_size]
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        probabilities = F.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
    return predicted_classes, probabilities
def evaluate_accuracy(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = torch.argmax(outputs, axis=1)
        accuracy = (predicted == torch.argmax(y_test, axis=1)).sum().item() / y_test.size(0)
    return accuracy

#train model

converted=converted/255
y_train = torch.from_numpy(y).long()
X_train, X_test, y_train, y_test = train_test_split(
    converted, y_train, test_size=0.2, random_state=42, shuffle=True
)

# Convert to torch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

classifier = ANN()
print("Classifier data shape:",converted.shape,y.shape)
train_model(classifier, X_train.float(), y_train.float(), num_epochs=200, lr=0.001)
test_acc = evaluate_accuracy(classifier, X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
torch.save(classifier.state_dict(), "/its/home/drs25/Tactile-Diffusion-Model/Data/models/classifier.pth")

##############################################
# Evaluate non linear
##############################################

