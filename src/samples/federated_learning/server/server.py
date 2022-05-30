import torch
from pathlib import Path
import os
from .model_aggregator import ModelAggregator
from .client_selector import ClientSelector

class Server():
    def __init__(self, config, shap_util):
        self.config = config
        self.default_model_path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        self.net = self.load_default_model()
        self.aggregator = ModelAggregator()
        self.selector = ClientSelector()
        self.shap_util = shap_util
        self.rounds = 0
        self.e = []
    
    def set_rounds(self, rounds):
        self.rounds = rounds
        
    def create_default_model(self):
        Path(os.path.dirname(self.default_model_path)).mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), self.default_model_path)
        print("default model saved to:{}".format(os.path.dirname(self.default_model_path)))
    
    def load_default_model(self):
        """
        Load a model from a file.
        """
        if not os.path.exists(self.default_model_path):
            self.create_default_model()
        if os.path.exists(self.default_model_path):
            try:
                model = self.config.NETWORK()
                model.load_state_dict(torch.load(self.default_model_path))
                model.eval()
                print("Load model successfully")
            except:
                print("Couldn't load model")
        else:
            print("Could not find model: {}".format(self.default_model_path))   
        return model
            
    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()
    
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(new_params, strict=True)
        
    def select_clients(self):
        return self.selector.random_selector(self.config.NUMBER_OF_CLIENTS, self.config.CLIENTS_PER_ROUND)

    def aggregate_model(self, client_parameters): 
        new_parameters = self.aggregator.model_avg(client_parameters)
        self.update_nn_parameters(new_parameters)
        if (self.rounds + 1)%50 == 0:
            print("Model aggregation in round {} was successful".format(self.rounds+1))
        
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