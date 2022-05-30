from .configuration import Configuration
from .dataset import Dataset
import torch
from numpy.random import default_rng


class ClientPlane():
    
    def __init__(self, config, observer_config, data, shap_util):
        """
        Simulation of isolated distributed clients
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        :param data: aggregated dataset 
        :type data: dataset.Dataset
        :param shap_util: utils for shap calculations
        :type shap_util: SHAPUtil
        """
        self.config = config
        self.observer_config = observer_config
        self.shap_util = shap_util
        self.train_dataset = data.train_dataset
        self.test_dataset = data.test_dataset
        self.train_dataloader = data.train_dataloader
        self.test_dataloader = data.test_dataloader
        self.ClientType = self.config.CLIENT_TYPE
        self.clients = self.create_clients()
        self.poisoned_clients = []
    
    def divide_data_equally(self):
        """
        Divides the dataset into NUMBER_OF_CLIENTS different subsets
        return torch.utils.data.Subset[]
        """
        indices = [[] for i in range(self.config.NUMBER_OF_CLIENTS)]
        for i in range(len(self.train_dataset)):
            indices[i % self.config.NUMBER_OF_CLIENTS].append(i)
        trainsets = [torch.utils.data.Subset(self.train_dataset, idx) for idx in indices]
        return trainsets

    def create_distributed_dataloaders(self, distributed_datasets):
        """
        Divides the dataset into NUMBER_OF_CLIENTS different subsets
        return torch.utils.data.DataLoader[]
        """
        dataloaders = [
            torch.utils.data.DataLoader(set, batch_size=self.config.BATCH_SIZE_TRAIN,shuffle=True, num_workers=2)
            for set in distributed_datasets
        ]
        return dataloaders
    
    def poison_clients(self):
        """
        Poison clients with selected poisoning attack 
        :TODO add different poisoning attacks
        :TODO poison subset of clients only
        """
        if self.config.DATA_POISONING_PERCENTAGE > 0:
            print("Flip {}% of the {} labels to {}".format(self.config.DATA_POISONING_PERCENTAGE * 100., self.config.FROM_LABEL, self.config.TO_LABEL))
            self.poisoned_clients = self.random_client_ids()
            for index, client_index in enumerate(self.poisoned_clients):
                if (index+1)%20 == 0:
                    print("{}/{} clients poisoned".format(index+1, len(self.poisoned_clients)))
                self.clients[client_index].label_flipping_data(from_label = self.config.FROM_LABEL, to_label = self.config.TO_LABEL, percentage = self.config.DATA_POISONING_PERCENTAGE)
        else: 
            print("No poisoning due to {}% poisoning rate".format(self.config.DATA_POISONING_PERCENTAGE * 100.))
            
    def random_client_ids(self): 
        rng = default_rng()
        return rng.choice(self.config.NUMBER_OF_CLIENTS, size=self.config.POISONED_CLIENTS, replace=False)
        

    def create_clients(self):
        """
        Create clients from dataloaders
        return Client[]
        """
        distributed_datasets = self.divide_data_equally()
        distributed_dataloaders = self.create_distributed_dataloaders(distributed_datasets)
        print("Create {} clients with dataset of size {}".format(self.config.NUMBER_OF_CLIENTS, len(distributed_dataloaders[0].dataset)))
        return [self.ClientType(self.config, self.observer_config, idx, dataloader, self.test_dataloader, self.shap_util) for idx, dataloader in enumerate(distributed_dataloaders)]
    
    def reset_client_nets(self):
        """
        Reset client's net to default
        """
        for index, client in enumerate(self.clients):
            client.reset_net()
        print("Reset networks successfully")
            
    def reset_poisoning_attack(self):
        for index, client in enumerate(self.clients):
            if (index+1)%20 == 0:
                print("{}/{} clients cleaned".format(index+1, len(self.clients)))
            client.reset_label_flipping_data(from_label=self.config.FROM_LABEL, percentage=self.config.DATA_POISONING_PERCENTAGE)
        print("Cleaning successfully")
        
    def update_config(self, config, observer_config):
        """
        Update client_plane configurations 
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        """
        self.config = config
        self.observer_config = observer_config
        self.ClientType = self.config.CLIENT_TYPE
        for index, client in enumerate(self.clients):
            client.update_config(config, observer_config)
        
            