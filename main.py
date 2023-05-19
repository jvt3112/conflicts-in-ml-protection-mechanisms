# defining a model (not same as Madry's paper)
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

class MNIST_CNN(nn.Module):
    def __init__(self,):
        super(MNIST_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10))

    def forward(self, x):
        out = self.net(x)
        return out
    


# apply_transform = transforms.Compose([
#             transforms.CenterCrop(28),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])])

apply_transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.ToTensor()])
norm = transforms.Normalize([0.5], [0.5])
data_dir = '../data/mnist/'

train_dataset = datasets.MNIST(data_dir, train=True, download=True,transform=apply_transform)

test_dataset = datasets.MNIST(data_dir, train=False, download=True,transform=apply_transform)

trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

def test_inference(model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(norm(images))
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference_adv(model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            images = attacker.perturb_2(images, labels)
            # Inference
            outputs = model(norm(images))
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add comments and docstrings. Finalise which perturbation method to use.
class PGDAttack:
    def __init__(self, model, epsilon, attack_steps, attack_lr, random_start=True):
        self.model = model
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.attack_lr = attack_lr
        self.rand = random_start
        self.clamp = (0,1)

    def random_init(self, x):
        x = x + (torch.rand_like(x) * 2 * self.epsilon - self.epsilon)
        x = torch.clamp(x,*self.clamp)
        return x

    def perturb(self, x, y):
        x_adv = x.detach().clone()

        if self.rand:
            x_adv = self.random_init(x_adv)

        for i in range(self.attack_steps):
            x_adv.requires_grad = True
            logits = self.model(norm(x_adv))
            self.model.zero_grad()
            
            loss = F.cross_entropy(logits, y,  reduction="sum")
            loss.backward()
            with torch.no_grad():                      
                grad = x_adv.grad
                grad = grad.sign()
                x_adv = x_adv + self.attack_lr * grad
                
                # Projection
                noise = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + noise, min=0, max=1)
        return x_adv
    
    def perturb_2(self, x, y):
        if self.rand:
            delta = torch.rand_like(x, requires_grad=True)
            delta.data = delta.data * 2 * self.epsilon - self.epsilon
        else:
            delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.attack_steps):
            loss = F.cross_entropy(self.model(norm(x + delta)), y)
            loss.backward()
            delta.data = (delta + self.attack_lr*delta.grad.detach().sign()).clamp(-self.epsilon,self.epsilon)
            delta.grad.zero_()
        return x+delta.detach()

# afterNORM + PERTURB_2
device = 'cuda'
global_model = MNIST_CNN()
global_model.to(device)
global_model.train()

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(global_model.parameters(), lr=0.005,momentum=0.9, weight_decay=5e-4)
attacker = PGDAttack(global_model, 0.3, 25, 0.01)

for iter in range(10):
    batch_loss = []
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        images = attacker.perturb_2(images, labels)

        optimizer.zero_grad()
        log_probs = global_model(norm(images))
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    if iter%2==0:
      print('Epoch:', iter)
      print('Total loss:', sum(batch_loss)/len(batch_loss))
      test_acc, test_loss = test_inference(global_model)
      adv_acc, adv_loss = test_inference_adv(global_model)
      print('Test loss:', test_loss)
      print('Test Accuracy:', test_acc)
      print('Adv Test loss:', adv_loss)
      print('Adv Test Accuracy:', adv_acc)
  
print('======FINISHED TRAINING======')
test_acc, test_loss = test_inference(global_model)
adv_acc, adv_loss = test_inference_adv(global_model)
print('Test loss:', test_loss)
print('Test Accuracy:', test_acc)
print('Adv Test loss:', adv_loss)
print('Adv Test Accuracy:', adv_acc)
