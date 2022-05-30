import torch, torchvision
from .dataloader import Dataloader

class MNISTDataloader(Dataloader): 
    
    def __init__(self, configs):
        super(MNISTDataloader, self).__init__(configs)
        
    def load_train_dataloader(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor()])
        
        train_dataset = torchvision.datasets.MNIST(
            self.configs.MNIST_DATASET_PATH, 
            train=True, download=True,
            transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.configs.BATCH_SIZE_TRAIN, 
            shuffle=False)
        
        print("MNIST training loader loaded.")
        return train_loader
    
    def load_test_dataloader(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.transforms.ToTensor()])
        
        test_dataset = torchvision.datasets.MNIST(
            self.configs.MNIST_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.configs.BATCH_SIZE_TEST, 
            shuffle=False)
        
        print("MNIST test loader loaded.")
        return test_loader
