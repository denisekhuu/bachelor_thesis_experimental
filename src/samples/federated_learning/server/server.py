import torch
import os
import copy

from .model_aggregator import ModelAggregator
from .client_selector import ClientSelector
from ..observer import ServerObserver
from ..utils import CNNHandler

class Server(CNNHandler):
    def __init__(self, config, observer_config, train_dataloader, test_dataloader, shap_util):
        
        super(Server, self).__init__(config, observer_config, train_dataloader, test_dataloader, shap_util)
        self.default_model_path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        self.observer = ServerObserver(config, observer_config)
        self.aggregator = ModelAggregator()
        self.selector = ClientSelector()
    
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
        self.net.eval()
        
    def select_clients(self):
        return self.selector.random_selector(self.config.NUMBER_OF_CLIENTS, self.config.CLIENTS_PER_ROUND)

    def aggregate_model(self, client_parameters): 
        new_parameters = self.aggregator.model_avg(client_parameters)
        self.update_nn_parameters(new_parameters)
        if (self.rounds + 1)%50 == 0:
            print("Model aggregation in round {} was successful".format(self.rounds+1))
        
    def update_config(self, config, observer_config):
        super().update_config(config, observer_config)
        self.default_model_path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))