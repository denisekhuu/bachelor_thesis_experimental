import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from ..observer import ClientObserver
import copy
import os

class Client(): 
    def __init__(self, config, observer_config, client_id, train_dataloader, test_dataloader, shap_util):
        """
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        :param client_id: client id
        :type observerconfig: int
        :param train_dataloader: Training data loader
        :type train_dataloader: torch.utils.data.DataLoader
        :param test_dataloader: Test data loader
        :type test_dataloader: torch.utils.data.DataLoader
        :param shap_util: utils for shap calculations
        :type shap_util: SHAPUtil
        """
        self.config = config
        self.observer_config = observer_config
        self.shap_util = shap_util
        self.net = self.load_default_model()
        self.observer = ClientObserver(self.config, self.observer_config, client_id, False, len(train_dataloader.dataset))
        self.client_id = client_id
        self.rounds = 0
        
        # datasets
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        # loss metrics
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(self.train_dataloader.dataset) for i in range(self.config.N_EPOCHS)]
        
        # raw performance measures
        self.confusion_matrix = torch.zeros(self.config.NUMBER_TARGETS, self.config.NUMBER_TARGETS)
        self.correct = 0
        
        # derived metrics
        self.accuracy = 0
        self.recall = []
        self.precision = []
        
        # SHAP utils
        self.e = None
        self.shap_values = []
        self.shap_prediction = []
        
        # SHAP metrics
        self.positive_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.negative_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.non_zero_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        
        # label flipping meta data
        self.is_poisoned = False
        self.poisoned_indices = []
        self.poisoning_indices = []
        
        self.load_default_model()
        
    def label_flipping_data(self, from_label, to_label, percentage=1): 
        """
        Label Flipping attack on distributed client 
        :param from_label: label to be flipped
        :type from_label: 
        :param to_label: label flipped to
        :typeto_label: 
        """
        indices = (self.train_dataloader.dataset.dataset.targets[self.train_dataloader.dataset.indices] == from_label).nonzero(as_tuple=False)
        last_index = int(len(indices) * percentage)
        self.poisoning_indices = indices
        self.poisoned_indices = indices if percentage == 1 else indices[:last_index]
        self.train_dataloader.dataset.dataset.targets[self.poisoned_indices] = to_label
        self.observer.set_poisoned(True)
        self.is_poisoned = True
        
    def reset_label_flipping_data(self,from_label, percentage=1):
        if self.is_poisoned: 
            self.train_dataloader.dataset.dataset.targets[self.poisoning_indices] = from_label
            self.is_poisoned = False
            self.observer.set_poisoned(False)
        
    def reset_to_default_net(self):
        """
        Set to untrained new model
        """
        self.net = self.config.NETWORK()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        
    def reset_net(self): 
        """
        Set model to previous default parameters
        """
        self.net.apply(self.weight_reset)
        
    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)
        
        
    def weight_reset(self, m):
        """
        Reset weights of model
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
        
    def get_shap_values(self):
        """
        Calculate SHAP values and SHAP image predictions 
        """
        if not self.e: 
            self.e = self.shap_util.set_deep_explainer(self.net)
        self.shap_values = self.shap_util.get_shap_values(self.e)
        self.shap_prediction = self.shap_util.predict(self.net)
        
    def set_explainer(self): 
        self.e = self.shap_util.deep_explainer(self.net)
          
    def plot_shap_values(self, file):
        """
        Plot SHAP images
        """
        self.shap_util.plot(self.shap_values, file)
        
    def analize_test(self):
        """
        Calculate test metrics like accuracy, precision and recall
        """
        self.recall = self.confusion_matrix.diag()/self.confusion_matrix.sum(1)
        self.precision = self.confusion_matrix.diag()/self.confusion_matrix.sum(0)
        self.accuracy = self.correct / len(self.test_dataloader.dataset)
        self.precision[torch.isnan(self.precision)] = 0
        self.recall[torch.isnan(self.recall)] = 0
        
        
    def analize_shap_values(self): 
        """
        Calculate SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        """
        for i in range(self.config.NUMBER_TARGETS):
            self.positive_shap[i] = [np.sum(np.array(arr) > 0) for arr in self.shap_values[i]]
            self.negative_shap[i] = [np.sum(np.array(arr) < 0) for arr in self.shap_values[i]]
            self.non_zero_mean[i] = [arr[np.nonzero(arr)].mean() for arr in self.shap_values[i]]
            
    def analize(self):
        """
        Calculate SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        and test metrics like accuracy, precision and recall
        """
        self.analize_test()
        self.get_shap_values()
        self.analize_shap_values()
    
    def push_metrics(self): 
        """
        Push SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        and test metrics like accuracy, precision and recall to victoria metrics
        """
        self.analize()
        self.observer.push_metrics(self.accuracy, self.recall, self.precision, self.positive_shap, self.negative_shap, self.non_zero_mean)
        
    def update_config(self, config, observer_config):
        """
        Update client configurations 
        :param config: experiment configurations
        :type config: Configuration
        :param observer_config: observer configurations
        :type observer_config: ObserverConfiguration
        """
        self.config = config
        self.observer_config = observer_config
        self.test_counter = [i*len(self.train_dataloader.dataset) for i in range(self.config.N_EPOCHS)]
        self.confusion_matrix = torch.zeros(self.config.NUMBER_TARGETS, self.config.NUMBER_TARGETS)
        self.positive_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.negative_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.non_zero_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.observer.update_config(config, observer_config)
        
    def set_rounds(self, rounds):
        self.rounds = rounds
        self.observer.set_rounds(rounds)
            
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
        
    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()
    
    def load_default_model(self):
        """
        Load a model from a file to achive a common default behavior
        """
        path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        if os.path.exists(path):
            try:
                model = self.config.NETWORK()
                model.load_state_dict(torch.load(path))
                model.eval()
            except:
                print("Couldn't load model")
        else:
            print("Could not find model: {}".format(self.default_model_path))
        return model