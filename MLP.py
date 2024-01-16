# implementation of multilayer perceptron
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkBase import NeuralNetworkBase

class MLP(NeuralNetworkBase):
    def __init__(self, input_size, hidden_layers, num_classes, batch_norm):
        #inherit from nn.module
        super(MLP, self).__init__()
    
        layers = self.create_layers(input_size, hidden_layers, num_classes, batch_norm)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x
    


    def create_layers(self, input_size, hidden_layers, num_classes, batch_norm):
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        #test: batch normalisierung
        if batch_norm == True:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))  # Batch-Normalization

        layers.append(nn.ReLU())

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        layers.append(nn.Softmax(dim=1))
        
        
        #layers.append(nn.Sigmoid())
        
        return layers
        
    
    def train_model(self, train_loader, val_loader, num_epochs, criterion, optimizer, validation):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            self.train()

            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for data, labels in train_loader:
                data = data.view(data.size(0), -1)
                output = self(data)

                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted_train = torch.max(output.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            average_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            train_losses.append(average_train_loss)
            train_accuracies.append(train_accuracy)

            if validation:
                self.eval()
                val_loss = 0.0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for val_data, val_labels in val_loader:
                        val_data = val_data.view(val_data.size(0), -1)
                        val_output = self(val_data)
                        val_loss += criterion(val_output, val_labels).item()

                        _, predicted_val = torch.max(val_output.data, 1)
                        total_val += val_labels.size(0)
                        correct_val += (predicted_val == val_labels).sum().item()

                average_val_loss = val_loss / len(val_loader)
                val_accuracy = correct_val / total_val

                val_losses.append(average_val_loss)
                val_accuracies.append(val_accuracy)

                print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        return train_losses, train_accuracies, val_losses, val_accuracies

        
            
    def test_model(self, test_loader):
        
        self.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
    
        print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")

        return accuracy, precision, recall, f1


    def classify_images(self, images):
        self.eval()
        with torch.no_grad():
            # Assuming 'classifier' takes flattened images as input
            inputs = images.view(images.size(0), -1)
            outputs = self(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)

        return predicted_labels.numpy()



    def show_misses(model, test_loader, test_set, class_names):
        model.eval()
        all_predictions = []
        all_labels = []
        miss_indices = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        for i in range(len(all_predictions)):
            if all_predictions[i] != all_labels[i]:
                miss_indices.append(i)

        if len(miss_indices) != 0:
            figure = plt.figure(figsize=(15, 5))
            cols, rows = 4, 2

            for i in range(1, min(cols * rows + 1, len(miss_indices) + 1)):
                index = miss_indices[i]
                img, label = test_set[index]

                figure.add_subplot(rows, cols, i)
                plt.title(f'Label: {class_names[all_labels[index]]} Predicted: {class_names[all_predictions[index]]}')
                plt.axis("off")
                if img.shape[0] == 3:  # CIFAR-10
                    plt.imshow(np.transpose(img, (1, 2, 0)))
                else:  # Assume MNIST
                    plt.imshow(img[0], cmap="gray")

            plt.show()

