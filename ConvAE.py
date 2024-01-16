import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np
from NeuralNetworkBase import NeuralNetworkBase


class ConvAE(NeuralNetworkBase):
    def __init__(self, num_channels, hidden_channels, num_classes, batch_norm, input_size):
        super(ConvAE, self).__init__()

        # Encoder layers
        self.num_channels = num_channels
        encoder_layers, decoder_layers = self.create_layers(num_channels, hidden_channels, batch_norm)

        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        # works for cifar


        self.latent = nn.Linear(hidden_channels[-1] * 4 * 4, hidden_channels[-1])

        
        # Decoder layers
        self.decoder = nn.Sequential(*decoder_layers)

        #self.classifier = nn.Linear(hidden_channels[-1], num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    
    def create_layers(self, num_channels, hidden_channels, batch_norm):
        encoder_layers = []
        encoder_layers.append(nn.Conv2d(num_channels, hidden_channels[0], kernel_size=3, stride=2, padding=1))

        if batch_norm:
            encoder_layers.append(nn.BatchNorm2d(hidden_channels[0]))
        encoder_layers.append(nn.ReLU())

        for i in range(len(hidden_channels) - 1):
            encoder_layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())

        decoder_layers = []
        for i in range(len(hidden_channels) - 1, 0, -1):
            decoder_layers.append(nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.ConvTranspose2d(hidden_channels[0], num_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        decoder_layers.append(nn.Sigmoid())

        return encoder_layers, decoder_layers

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)

        encoded = encoded.view(encoded.size(0), -1)

        # Latent space (test)
        latent_space = self.latent(encoded)

        # Decoder


        decoded = self.decoder(encoded.view(encoded.size(0), -1, 4, 4))  # Adjust size based on your input size
  
            

        classifier = self.classifier(latent_space)

        return latent_space, decoded, classifier


    
    
    def test_model(self, test_loader, mode):
        
        
        if mode == "classify":
            self.eval()
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for data, labels in test_loader:
                    #data = data.view(data.size(0), -1)
                    if self.num_channels == 3:
                        data = data.view(-1, self.num_channels, 32, 32)  # Adjust based on your input size
                    elif self.num_channels == 1:
                        data = data.view(-1, self.num_channels, 28, 28)

                    _, _, classifications = self(data)

                    _, predicted_labels = torch.max(classifications, 1)
                    all_labels.extend(labels.numpy())
                    all_predictions.extend(predicted_labels.numpy())
                    
                    
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")
            
            return accuracy, precision, recall, f1    
        
        elif mode == "decode":
            self.eval()
            all_reconstructions = []
            all_originals = []

            with torch.no_grad():
                for data, _ in test_loader:
                    #data = data.view(data.size(0), -1)


                    if self.num_channels == 3:
                        data = data.view(-1, self.num_channels, 32, 32)  # Adjust based on your input size
                    elif self.num_channels == 1:
                        data = data.view(-1, self.num_channels, 28, 28)


                    _, reconstructions, _ = self(data)
                    all_reconstructions.append(reconstructions)
                    all_originals.append(data)

            # Check if lists are non-empty before attempting to concatenate
            if all_reconstructions and all_originals:
                all_reconstructions = torch.cat(all_reconstructions, dim=0).numpy()
                all_originals = torch.cat(all_originals, dim=0).numpy()

                mse = mean_squared_error(all_originals.flatten(), all_reconstructions.flatten())
                print(f'Mean Squared Error (MSE): {mse}')
            else:
                print("No data for concatenation.")

            return mse
            
        else:
            return None
        
        
        
        
    def train_model(self, data_loader, val_loader, num_epochs, num_channels, optimizer, criterion, validation, mode):
        
        
        if mode == "classify":
          
            self.train()
            train_outputs = []
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []

            # Trainingsschleife
            for epoch in range(num_epochs):
                self.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for data, labels in data_loader:
                    if num_channels == 3:
                        data = data.view(-1, num_channels, 32, 32)  # Adjust based on your input size
                    elif num_channels == 1:
                        data = data.view(-1, num_channels, 28, 28)

                    optimizer.zero_grad()
                    _, _, classifications = self(data)

                    loss = criterion(classifications, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    _, predicted = torch.max(classifications.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                train_loss = running_loss / len(data_loader)
                train_accuracy = correct_train / total_train

                train_outputs.append((num_epochs, data, classifications))
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                if validation == True:
                    # Validation
                    self.eval()  # Setze das Modell in den Evaluierungsmodus
                    val_loss = 0.0
                    correct_val = 0
                    total_val = 0

                    with torch.no_grad():
                        for val_data, val_labels in val_loader:
                            if self.num_channels == 3:
                                val_data = val_data.view(-1, self.num_channels, 32, 32)  # Adjust based on your input size
                            elif self.num_channels == 1:
                                val_data = val_data.view(-1, self.num_channels, 28, 28)

                            _, _, val_classifications = self(val_data)

                            loss = criterion(val_classifications, val_labels)
                            val_loss += loss.item()

                            _, predicted_val = torch.max(val_classifications.data, 1)
                            total_val += val_labels.size(0)
                            correct_val += (predicted_val == val_labels).sum().item()

                    avg_val_loss = val_loss / len(val_loader)
                    val_accuracy = correct_val / total_val

                    val_losses.append(avg_val_loss)
                    val_accuracies.append(val_accuracy)

                # Gib den Durchschnittsverlust für Trainings- und Validierungssets aus
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Training Loss: {train_loss:.4f}, '
                    f'Training Accuracy: {train_accuracy :.2f}, ', end=" ")

                if validation == True:
                    print(f'Validation Loss: {avg_val_loss:.4f}, '
                        f'Validation Accuracy: {val_accuracy:.2f}')
                else:
                    print()

            return train_losses, train_accuracies, val_losses, val_accuracies

          
          
            
        elif mode == "decode":
            
            self.train()
            train_outputs = []
            train_losses = []
            val_losses = []
            # Trainingsschleife
            for epoch in range(num_epochs):
                self.train()
                running_loss = 0.0
                for data, labels in data_loader:
                    if num_channels == 3:
                        data = data.view(-1, num_channels, 32, 32)  # Anpassung der Dimensionen für CIFAR-10
                    elif num_channels == 1:
                        data = data.view(-1, num_channels, 28, 28)

            
                    optimizer.zero_grad()
                    _, reconstructed, _ = self(data)
                    

                    loss = criterion(reconstructed, data)    
                        
                    loss.backward()
                    optimizer.step()  
                    running_loss += loss.item()

                    # Vorwärtsdurchlauf         

                average_train_loss = running_loss / len(data_loader)
                train_outputs.append((epoch, data, reconstructed)) 
                    
                train_losses.append(average_train_loss)
                    
                # validation
            
                if validation == True:
                    val_loss = 0.0
                    
                    self.eval()  # Setze das Modell in den Evaluierungsmodus
                    with torch.no_grad():
                        for val_data, val_labels in val_loader:
                            #val_data = val_data.view(-1, input_size)
                            
                            if num_channels == 3:
                                val_data = data.view(-1, num_channels, 32, 32)  # Anpassung der Dimensionen für CIFAR-10
                            elif num_channels == 1:
                                val_data = data.view(-1, num_channels, 28, 28)

                            
                            _, val_reconstructed, _ = self(val_data)

                            val_loss += criterion(val_reconstructed, val_data).item()


                        # Berechne den Durchschnittsverlust für Trainings- und Validierungssets
                        avg_val_loss = val_loss / len(val_loader)       
                        val_losses.append(avg_val_loss)
                    
                
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Training Loss: {loss.item():.4f}, ', end = " ")
                if validation == True:
                    print(f'Validation Loss: {avg_val_loss:.4f}')  
                else:
                    print()
                    
            return train_losses, val_losses, train_outputs     
            
        else:
            return None
             

    
    def classify_images(self, images):
        self.eval()
        with torch.no_grad():
            # Assuming 'classifier' takes flattened images as input
            inputs = images
            _, _, outputs = self(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)

        return predicted_labels.numpy()
    
    
    def show_misses(self, test_loader, test_set, classes):
    
        self.eval()
        all_predictions = []
        all_labels = []
        
        miss_indices = []
        

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.view(data.size(0), -1)
                _, _, classifications = self(data)

                _, predicted_labels = torch.max(classifications, 1)
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted_labels.numpy())
          

        for i in range(len(all_predictions)):
            if all_predictions[i] != all_labels[i]:
                miss_indices.append(i)
            
            
        if len(miss_indices) != 0:
            
            figure = plt.figure(figsize=(15, 7))
            cols, rows = 4, 2
                    
            for i in range(1, min(cols*rows+1, len(miss_indices)+1)):
                
                index = miss_indices[i]
                img, label = test_set[index]

                figure.add_subplot(rows, cols, i)
                img = np.transpose(img, (1, 2, 0))
                plt.title(f'Label: {classes[label]} \nPredicted: {classes[all_predictions[index]]}')
                plt.axis("off")
                
                # check if mnist or cifar dataset. bad coding. just for testing purpose. will be removed later.
                if classes[0] == "plane":
                   plt.imshow(img)    
                else:             
                    plt.imshow(img, cmap = "gray")
            plt.show()
        
    def encode(self, x):
        # Nur den Encoder verwenden, um Daten zu codieren
        x = self.encoder(x)
        return x, None

    def decode(self, x):
        # Nur den Decoder verwenden, um Daten zu dekodieren
        x = self.decoder(x)
        return x, None

