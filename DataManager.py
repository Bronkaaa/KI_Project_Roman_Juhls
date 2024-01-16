import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from sklearn.utils import shuffle


# class for loading and managing datasets

class DataSetConfig():
    def __init__(self):
        
        #properties of dataset
        self.index = None
        
        self.classes = None
        self.num_classes = None
        self.dimensions = None
        self.num_channels = None


        # datasets and loaders for unnormalized dataset
        self.train_set = None
        self.train_loader = None
        self.test_set = None
        self.test_loader = None
        self.val_set = None
        self.val_set_norm = None
          
        # subset and loader for unnormalized dataset
        self.train_subset = None
        self.train_subset_loader = None
        
        
        # datasets and loaders for normalized dataset
        self.val_loader = None
        self.val_loader_norm = None
        
        self.train_set_norm = None
        self.train_loader_norm  = None
        self.test_set_norm  = None
        self.test_loader_norm  = None
          
        # subset and loader for normalized dataset
        self.train_subset_norm  = None
        self.train_subset_loader_norm  = None     
        
        

                        
        


class DataManagerSingleton():
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManagerSingleton, cls).__new__(cls)
            # Initialisierung des DataManager
            cls._instance._initialize_data_manager()
        return cls._instance
        
    def _initialize_data_manager(self):
        
        
        #variables for CIFAR dataset
        self.dataset_CIFAR = DataSetConfig()
        
        # augmented CIFAR-10 dataset
        self.dataset_CIFAR_augmented = DataSetConfig()
        
        #variables for MNIST dataset
        self.dataset_MNIST = DataSetConfig()
     
            
            
    def load_CIFAR10_dataset_augmented(self, batch_size):
        


        
        transform = transforms.ToTensor()
        transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        ])
        
        root = "./data"
        
        transform_norm = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        transform_augmented_norm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
              #transform PIL images to Tensors and normalize
        root_norm = "./data_normalized"

            
        trainset_reg = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        
        trainset_reg_norm = torchvision.datasets.CIFAR10(root=root_norm, train=True,
                                download=True, transform=transform_norm)   

         
        trainset_augmented = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform_augmented)
        
        trainset_augmented_norm = torchvision.datasets.CIFAR10(root=root_norm, train=True,
                                download=True, transform=transform_augmented_norm)   


        trainset = ConcatDataset([trainset_reg, trainset_augmented])
        trainset = shuffle(trainset, random_state=42)  # Shuffle the combined dataset

        trainset_norm = ConcatDataset([trainset_reg_norm, trainset_augmented_norm])
        trainset_norm = shuffle(trainset_norm, random_state=42)  # Shuffle the combined dataset

        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                    shuffle=True, num_workers=2)

        trainloader_norm = torch.utils.data.DataLoader(trainset_norm, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        
        
        
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainSubset, val_set = torch.utils.data.random_split(trainset, [train_size, val_size])
        
        trainSubsetloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        valLoader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
        
        trainSubset_norm, val_set_norm = torch.utils.data.random_split(trainset_norm, [train_size, val_size])
        
        trainSubsetloader_norm = torch.utils.data.DataLoader(trainSubset_norm, batch_size=batch_size, shuffle=False, num_workers=2)  
        
        valLoader_norm = torch.utils.data.DataLoader(val_set_norm, batch_size=batch_size, shuffle=False, num_workers=2)  
        
        
        
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)   

        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
               
       
        testset_norm = torchvision.datasets.CIFAR10(root=root_norm, train=False,
                                    download=True, transform=transform_norm)   

        testloader_norm = torch.utils.data.DataLoader(testset_norm, batch_size=batch_size,
                                        shuffle=False, num_workers=2)
        


        #general attributes
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.dataset_CIFAR_augmented.classes = classes
        self.dataset_CIFAR_augmented.num_classes = len(classes)
        self.dataset_CIFAR_augmented.num_channels = 3
        # calculate the number of input features by multiplying the dimensions
        self.dataset_CIFAR_augmented.dimensions = 3*32*32      
        
          
        ## trainset and trainloader
        self.dataset_CIFAR_augmented.train_set = trainset
        self.dataset_CIFAR_augmented.train_loader = trainloader
        
        #trainset normalized and trainloader normalized
        self.dataset_CIFAR_augmented.train_set_norm = trainset_norm
        self.dataset_CIFAR_augmented.train_loader_norm = trainloader_norm
        
        
        ## train subset and val set
        self.dataset_CIFAR_augmented.train_subset = trainSubset
        self.dataset_CIFAR_augmented.val_set = val_set     
        
        
        # trainSubset loader and validation loader
        self.dataset_CIFAR_augmented.train_subset_loader = trainSubsetloader        
        self.dataset_CIFAR_augmented.val_loader = valLoader
        
        # trainSubset Normalized and validation Set Normalized        
        self.dataset_CIFAR_augmented.train_subset_norm = trainSubset_norm     
        self.dataset_CIFAR_augmented.val_set_norm = val_set_norm
                
        # trainSubset Loader Normalized  and Validation Laoder Normalized
        
        self.dataset_CIFAR_augmented.train_subset_loader_norm = trainSubsetloader_norm        
        self.dataset_CIFAR_augmented.val_loader_norm = valLoader_norm        
    
        
        # trainset and testset sets and loaders unnormalized 

        self.dataset_CIFAR_augmented.test_set = testset
        self.dataset_CIFAR_augmented.test_loader = testloader        
        
        # trainset and testset sets and loaders normalized 
        self.dataset_CIFAR_augmented.test_set_norm = testset_norm
        self.dataset_CIFAR_augmented.test_loader_norm = testloader_norm 
        
        
        
        
    # load dataset    
    def load_CIFAR10_dataset(self, batch_size):
        
        #transform PIL images to Tensors and normalize
        

        transform = transforms.ToTensor()
        root = "./data"
        

        transform_norm = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        root_norm = "./data_normalized"

            
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
        
        trainset_norm = torchvision.datasets.CIFAR10(root=root_norm, train=True,
                                        download=True, transform=transform_norm)

        trainloader_norm = torch.utils.data.DataLoader(trainset_norm, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
        
    
        
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainSubset, val_set = torch.utils.data.random_split(trainset, [train_size, val_size])
        
        trainSubsetloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        valLoader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
        
        trainSubset_norm, val_set_norm = torch.utils.data.random_split(trainset_norm, [train_size, val_size])
        
        trainSubsetloader_norm = torch.utils.data.DataLoader(trainSubset_norm, batch_size=batch_size, shuffle=False, num_workers=2)  
        
        valLoader_norm = torch.utils.data.DataLoader(val_set_norm, batch_size=batch_size, shuffle=False, num_workers=2)  
        
        
        
        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                            download=True, transform=transform)   

        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
               
       
        testset_norm = torchvision.datasets.CIFAR10(root=root_norm, train=False,
                                    download=True, transform=transform_norm)   

        testloader_norm = torch.utils.data.DataLoader(testset_norm, batch_size=batch_size,
                                        shuffle=False, num_workers=2)
        


        #general attributes
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.dataset_CIFAR.classes = classes
        self.dataset_CIFAR.num_classes = len(classes)
        self.dataset_CIFAR.num_channels = 3
        # calculate the number of input features by multiplying the dimensions
        self.dataset_CIFAR.dimensions = 3*32*32      
        
          
        ## trainset and trainloader
        self.dataset_CIFAR.train_set = trainset
        self.dataset_CIFAR.train_loader = trainloader
        
        #trainset normalized and trainloader normalized
        self.dataset_CIFAR.train_set_norm = trainset_norm
        self.dataset_CIFAR.train_loader_norm = trainloader_norm
        
        
        ## train subset and val set
        self.dataset_CIFAR.train_subset = trainSubset
        self.dataset_CIFAR.val_set = val_set     
        
        
        # trainSubset loader and validation loader
        self.dataset_CIFAR.train_subset_loader = trainSubsetloader        
        self.dataset_CIFAR.val_loader = valLoader
        
        # trainSubset Normalized and validation Set Normalized        
        self.dataset_CIFAR.train_subset_norm = trainSubset_norm     
        self.dataset_CIFAR.val_set_norm = val_set_norm
                
        # trainSubset Loader Normalized  and Validation Laoder Normalized
        
        self.dataset_CIFAR.train_subset_loader_norm = trainSubsetloader_norm        
        self.dataset_CIFAR.val_loader_norm = valLoader_norm        
    
        
        # trainset and testset sets and loaders unnormalized 

        self.dataset_CIFAR.test_set = testset
        self.dataset_CIFAR.test_loader = testloader        
        
        # trainset and testset sets and loaders normalized 
        self.dataset_CIFAR.test_set_norm = testset_norm
        self.dataset_CIFAR.test_loader_norm = testloader_norm 
 
        
    def load_MNIST_dataset(self, batch_size):
   

        transform = transforms.ToTensor()
        root = "./data"

        transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        root_norm = "./data_normalized"



        trainset = torchvision.datasets.MNIST(root=root, train=True,
                                                download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, num_workers=2)    
        
        
        
        
        trainset_norm = torchvision.datasets.MNIST(root=root_norm, train=True,
                                        download=True, transform=transform_norm)
        
        trainloader_norm = torch.utils.data.DataLoader(trainset_norm, batch_size=batch_size,
                                shuffle=True, num_workers=2)



        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainSubset, val_set = torch.utils.data.random_split(trainset, [train_size, val_size])
        
        
        trainSubsetloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        valLoader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)        
        
        
        
        trainSubset_norm, val_set_norm = torch.utils.data.random_split(trainset_norm, [train_size, val_size])
     
        trainSubsetloader_norm = torch.utils.data.DataLoader(trainSubset_norm, batch_size=batch_size, shuffle=False, num_workers=2)
        
        valLoader_norm = torch.utils.data.DataLoader(val_set_norm, batch_size=batch_size, shuffle=False, num_workers=2)
        
        
        
        testset = torchvision.datasets.MNIST(root=root, train=False,
                                            download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)
        
        
        
        testset_norm = torchvision.datasets.MNIST(root=root_norm, train=False,
                                        download=True, transform=transform_norm)
        

        testloader_norm = torch.utils.data.DataLoader(testset_norm, batch_size=batch_size,
                                                shuffle=False, num_workers=2)



        classes = ('0', '1', '2','3', '4','5', '6','7', '8','9')
        
        self.dataset_MNIST.classes = classes
        self.dataset_MNIST.num_classes = len(classes)
        self.dataset_MNIST.num_channels = 1
        ## calculate dimensions of MNIST
        self.dataset_MNIST.dimensions = 28*28        
        
        
        
        
        ## trainset and trainloader
        self.dataset_MNIST.train_set = trainset
        self.dataset_MNIST.train_loader = trainloader
        
        #trainset normalized and trainloader normalized
        self.dataset_MNIST.train_set_norm = trainset_norm
        self.dataset_MNIST.train_loader_norm = trainloader_norm
        
        
        ## train subset and val set
        self.dataset_MNIST.train_subset = trainSubset
        self.dataset_MNIST.val_set = val_set     
        
        
        # trainSubset loader and validation loader
        self.dataset_MNIST.train_subset_loader = trainSubsetloader        
        self.dataset_MNIST.val_loader = valLoader
        
        # trainSubset Normalized and validation Set Normalized        
        self.dataset_MNIST.train_subset_norm = trainSubset_norm     
        self.dataset_MNIST.val_set_norm = val_set_norm
                
        # trainSubset Loader Normalized  and Validation Laoder Normalized
        
        self.dataset_MNIST.train_subset_loader_norm = trainSubsetloader_norm        
        self.dataset_MNIST.val_loader_norm = valLoader_norm        
    
        
        # trainset and testset sets and loaders unnormalized 

        self.dataset_MNIST.test_set = testset
        self.dataset_MNIST.test_loader = testloader        
        
        # trainset and testset sets and loaders normalized 
        self.dataset_MNIST.test_set_norm = testset_norm
        self.dataset_MNIST.test_loader_norm = testloader_norm 
        