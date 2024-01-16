# implementation of a Deep Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from NeuralNetworkBase import NeuralNetworkBase

    
class DeepCNN(NeuralNetworkBase):
    def __init__(self, num_channels, num_classes, batch_norm):
        super(DeepCNN, self).__init__()
        # Convolutional layers
        self.batch_norm = batch_norm
        
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Fully connected layers
        self.flatten = nn.Flatten()
        

        if num_channels == 3:
            self.fc1 = nn.Linear(128 * 4 * 4, 128)
        if num_channels == 1:
            self.fc1 = nn.Linear(128 * 3 * 3, 128)


        
        self.dropout4 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layers
        
        x = self.conv1(x)
        if self.batch_norm:
            x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.batch_norm2(x)
        x = F.relu(x)
        
        x = self.pool1(x)
        x = self.dropout1(x)
        
        
        x = self.conv3(x)
        if self.batch_norm:
            x = self.batch_norm3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        if self.batch_norm:
            x = self.batch_norm4(x)
        x = F.relu(x)
        
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv5(x)
        if self.batch_norm:
            x = self.batch_norm5(x)
        x = F.relu(x)
        
        x = self.conv6(x)
        if self.batch_norm:    
            x = self.batch_norm6(x)
        x = F.relu(x)
        
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.dropout4(F.relu(self.fc1(self.flatten(x))))
        x = self.fc2(x)

        """
        x = self.dropout1(self.pool1(F.relu(self.batch_norm2(self.conv2(F.relu(self.batch_norm1(self.conv1(x))))))))
        x = self.dropout2(self.pool2(F.relu(self.batch_norm4(self.conv4(F.relu(self.batch_norm3(self.conv3(x))))))))
        x = self.dropout3(self.pool3(F.relu(self.batch_norm6(self.conv6(F.relu(self.batch_norm5(self.conv5(x))))))))

        # Fully connected layers
        x = self.dropout4(F.relu(self.fc1(self.flatten(x))))
        x = self.fc2(x)
        """
        
        return x


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

