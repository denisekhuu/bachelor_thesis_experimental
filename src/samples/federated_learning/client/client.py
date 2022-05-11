import torch
import torch.optim as optim
import torch.nn as nn

class Client(): 
    def __init__(self, configs, train_dataloader, test_dataloader):
        """
        :param configs: experiment configurations
        :type configs: Configuration
        :param train_dataloader: Training data loader
        :type train_dataloader: torch.utils.data.DataLoader
        :param test_dataloader: Test data loader
        :type test_dataloader: torch.utils.data.DataLoader
        """
        self.configs = configs
        self.net = self.configs.NETWORK()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.configs.LEARNING_RATE, momentum=self.configs.MOMENTUM)
        self.poisoned = True
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(self.train_dataloader.dataset) for i in range(self.configs.N_EPOCHS)]
        
    def label_flipping_data(self, from_label, to_label): 
        """
        Label Flipping attack on distributed client 
        :param from_label: label to be flipped
        :type from_label: 
        :param to_label: label flipped to
        :typeto_label: 
        """
        self.train_dataloader.dataset.dataset.targets = torch.where(self.train_dataloader.dataset.dataset.targets == from_label, to_label, self.train_dataloader.dataset.dataset.targets)
        print("Label Flipping from {} to {}".format(from_label, to_label))