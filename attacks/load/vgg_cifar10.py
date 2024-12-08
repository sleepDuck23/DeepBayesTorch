import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class CIFAR10VGG(nn.Module):
    def __init__(self, num_classes=10, weight_decay=0.0005):
        super(CIFAR10VGG, self).__init__()
        
        def conv_block(in_channels, out_channels, dropout_rate=0, weight_decay=weight_decay):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(dropout_rate)
            )
        
        self.features = nn.Sequential(
            # Block 1
            conv_block(3, 64, dropout_rate=0.3),
            conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            conv_block(64, 128, dropout_rate=0.4),
            conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            conv_block(128, 256, dropout_rate=0.4),
            conv_block(256, 256, dropout_rate=0.4),
            conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            conv_block(256, 512, dropout_rate=0.4),
            conv_block(512, 512, dropout_rate=0.4),
            conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            conv_block(512, 512, dropout_rate=0.4),
            conv_block(512, 512, dropout_rate=0.4),
            conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return model

def load_cifar10_data(data_dir='./cifar10_data', batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == '__main__':
    data_dir = './cifar10_data'
    dataloaders = load_cifar10_data(data_dir)
    
    model = CIFAR10VGG()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=250)
    
    torch.save(model.state_dict(), 'cifar10vgg.pth')
    print("Model training complete.")