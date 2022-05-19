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
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_accuracy = 0
        self.test_counter = [i*len(self.train_dataloader.dataset) for i in range(self.configs.N_EPOCHS)]
        self.confusion_matrix = torch.zeros(self.configs.NUMBER_TARGETS, self.configs.NUMBER_TARGETS)
        self.target_accuracy = []
        
    def label_flipping_data(self, from_label, to_label, percentage = 1): 
        """
        Label Flipping attack on distributed client 
        :param from_label: label to be flipped
        :type from_label: 
        :param to_label: label flipped to
        :typeto_label: 
        """
        indices = (self.train_dataloader.dataset.dataset.targets == from_label).nonzero(as_tuple=False)
        last_index = int(len(indices) * percentage)
        self.poisoned_indices = indices if percentage == 1 else indices[:last_index]
        self.train_dataloader.dataset.dataset.targets[self.poisoned_indices] = to_label
            
        print("Label Flipping {}% from {} to {}".format(100. * percentage, from_label, to_label))
        
    def set_net(self):
        """
        Set to untrained model net 
        """
        self.net = self.configs.NETWORK()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.configs.LEARNING_RATE, momentum=self.configs.MOMENTUM)
        
    def reset_net(self): 
        """
        Set model to previous default parameters
        """
        self.net.apply(self.weight_reset)
        
    def weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()