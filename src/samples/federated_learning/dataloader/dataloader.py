from abc import abstractmethod
class Dataloader(): 
    
    def __init__(self, configs):
        self.configs = configs
        self.train_dataloader = self.load_train_dataloader()
        self.test_dataloader = self.load_test_dataloader()
    
    @abstractmethod
    def load_train_dataloader(self):
        """
        Loads & returns the training dataloader.

        :return: torchvision.Dataloader
        """
        raise NotImplementedError("load_train_dataloader() isn't implemented")
        
    @abstractmethod
    def load_test_dataloader(self):
        """
        Loads & returns the test dataloader.

        :return:torchvision.Dataloader
        """
        raise NotImplementedError("load_test_dataloader() isn't implemented")