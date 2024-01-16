import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class NeuralNetworkBase(ABC, nn.Module):
    def __init__(self):
        super(NeuralNetworkBase, self).__init__()


    @abstractmethod
    def train_model():
        pass
    
    @abstractmethod
    def test_model():
        pass
    
    
    def save_model(self, file_path):
        # Speichere den Zustand des Modells
        torch.save(self.state_dict(), file_path)
        print(f'Modell wurde unter {file_path} gespeichert.')

    def load_model(self, file_path):
        # Lade den Zustand des Modells aus der Datei
        self.load_state_dict(torch.load(file_path))
        print(f'Modell wurde aus {file_path} geladen.')
        