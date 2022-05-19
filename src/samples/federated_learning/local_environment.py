from .configuration import Configuration
from .dataset import Dataset
import torch

class LocalEnvironment():
    
    def __init__(self, configs:  Configuration, data: Dataset):
        """
        Simulation of isolated distributed clients
        :param configs: experiment configurations
        :type configs: Configuration
        :param data: aggregated dataset 
        :type configs: dataset.Dataset
        """
        self.configs = configs
        self.train_dataset = data.train_dataset
        self.test_dataset = data.test_dataset
        self.train_dataloader = data.train_dataloader
        self.test_dataloader = data.test_dataloader
        self.ClientType = self.configs.CLIENT_TYPE
        self.clients = self.create_clients()
    
    def divide_data_equally(self):
        """
        Divides the dataset into NUMBER_OF_CLIENTS different subsets
        return torch.utils.data.Subset[]
        """
        indices = [[] for i in range(self.configs.NUMBER_OF_CLIENTS)]
        for i in range(len(self.train_dataset)):
            indices[i % self.configs.NUMBER_OF_CLIENTS].append(i)
        trainsets = [torch.utils.data.Subset(self.train_dataset, idx) for idx in indices]
        return trainsets

    def create_distributed_dataloaders(self, distributed_datasets):
        """
        Divides the dataset into NUMBER_OF_CLIENTS different subsets
        return torch.utils.data.DataLoader[]
        """
        dataloaders = [
            torch.utils.data.DataLoader(set, batch_size=self.configs.BATCH_SIZE_TRAIN,shuffle=True, num_workers=2)
            for set in distributed_datasets
        ]
        return dataloaders
    
    def poison_clients(self):
        """
        Poison Clients with selected poisoning attack 
        :TODO add different poisoning attacks
        :TODO poison subset of clients only
        """
        for index, client in enumerate(self.clients): 
            print("{}/{} clients poisoned".format(index+1, len(self.clients)))
            client.label_flipping_data(from_label = self.configs.FROM_LABEL, to_label = self.configs.TO_LABEL, percentage = self.configs.DATA_POISONING_PERCENTAGE)
            

    def create_clients(self):
        """
        Create clients from dataloaders
        return Client[]
        """
        distributed_datasets = self.divide_data_equally()
        distributed_dataloaders = self.create_distributed_dataloaders(distributed_datasets)
        print("Create {} clients".format(self.configs.NUMBER_OF_CLIENTS))
        return [self.ClientType(self.configs, dataloader, self.test_dataloader) for dataloader in distributed_dataloaders]
    
    def reset_client_nets(self):
        """
        Reset client's net to default
        """
        for index, client in enumerate(self.clients):
            client.reset_net()
            print("Reset network successfully")