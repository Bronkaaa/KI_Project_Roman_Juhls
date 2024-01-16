# implementation of a Deep Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from NeuralNetworkBase import NeuralNetworkBase


# Create a neural net class
class CNN(NeuralNetworkBase):
    
    
    # Defining the Constructor
    def __init__(self, num_channels, fully_connected_layer = 128, batch_norm=True):
        super(CNN, self).__init__()
        

        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        if batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)




        self.flatten = nn.Flatten()
        
        self.fc1 = None
        #CIFAR: 
        if num_channels == 3:
            self.fc1 = nn.Linear(64 * 8 * 8, fully_connected_layer)
        
        #MNIST:
        elif num_channels == 1:
            self.fc1 = nn.Linear(64 * 7 * 7, fully_connected_layer)

        if batch_norm:
            self.batch_norm_fc = nn.BatchNorm1d(fully_connected_layer)
        
        self.fc2 = nn.Linear(fully_connected_layer, 10)
        
        
    def forward(self, x): 

        x = self.conv1(x)
        
        if hasattr(self, 'batch_norm1'):
            x = self.batch_norm1(x)

        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        
        if hasattr(self, 'batch_norm2'):
            x = self.batch_norm2(x)

        x = F.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        
        x = self.fc1(x)
        
        if hasattr(self, 'batch_norm_fc'):
            x = self.batch_norm_fc(x)

        x = F.relu(x)
        x = self.fc2(x)

        return x
    
    def create_layers(self, input_size, num_classes, num_channels):
    
        layers = []
        
        layers.append(nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU()) 
        layers.append(nn.Dropout(0.3))
        
        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.ReLU()) 
        layers.append(nn.Dropout(0.3))


        layers.append(nn.ReLU()) 

        return layers
    
    
    def train_model(self, train_loader, val_loader, num_epochs, criterion, optimizer, validation):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        self.train()

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            if validation:
                # Validation phase
                self.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for val_data, val_labels in val_loader:
                        val_outputs = self(val_data)
                        val_loss += criterion(val_outputs, val_labels).item()

                        _, predicted_val = torch.max(val_outputs.data, 1)
                        total_val += val_labels.size(0)
                        correct_val += (predicted_val == val_labels).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct_val / total_val

                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)

                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, '
                    f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            else:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        return train_losses, train_accuracies, val_losses, val_accuracies

        
        
    def test_model(self, test_loader):
        
        self.eval()
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = self(data)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.numpy())               

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
    
        print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")
        
        return accuracy, precision, recall, f1
        

    def classify_images(self, images):
        self.eval()
        with torch.no_grad():
            # Assuming 'classifier' takes images in the shape (batch_size, channels, height, width)
            outputs = self(images)
            _, predicted_labels = torch.max(outputs.data, 1)

        return predicted_labels.numpy()

