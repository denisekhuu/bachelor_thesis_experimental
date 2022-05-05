from abc import abstractmethod
class Dataset(): 
    
    def __init__(self, configs):
        self.configs = configs
        self.train_dataloader = self.load_train_dataloader()
        self.test_dataloader = self.load_test_dataloader()
    
    @abstractmethod
    def load_train_data(self):
        """
        Loads & returns the training dataloader and dataset.

        :return: torchvision.Dataloader, torchvision.Dataset
        """
        raise NotImplementedError("load_train_dataloader() isn't implemented")
        
    @abstractmethod
    def load_test_data(self):
        """
        Loads & returns the test dataloader and dataset. 

        :return:torchvision.Dataloader, torchvision.Dataset
        """
        raise NotImplementedError("load_test_dataloader() isn't implemented")