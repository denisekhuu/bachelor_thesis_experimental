import torch, torchvision
from .dataloader import Dataloader

class FashionMNISTDataloader(Dataloader): 
    
    def __init__(self, configs):
        super(FashionMNISTDataloader, self).__init__(configs)
        self.labels = self.configs.MNIST_FASHION_LABELS
        
    def load_train_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            self.configs.MNIST_FASHION_DATASET_PATH, 
            train=True, download=True,
            transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.configs.BATCH_SIZE_TRAIN, 
            shuffle=True)
        
        
        print("FashionMnist training loader loaded.")
        return train_loader
    
    def load_test_dataloader(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.FashionMNIST(
            self.configs.MNIST_FASHION_DATASET_PATH, 
            train=False, download=True,
            transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.configs.BATCH_SIZE_TEST, 
            shuffle=False)
        
        
        print("FashionMnist training loader loaded.")
        return test_loader
