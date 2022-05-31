import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import os
from .model_aggregator import ModelAggregator
from .client_selector import ClientSelector
import numpy as np

class Server():
    def __init__(self, config, test_loader, shap_util):
        self.config = config
        self.default_model_path = os.path.join(self.config.TEMP, 'models', "{}.model".format(self.config.MODELNAME))
        self.net = self.load_default_model()
        self.test_dataloader = test_loader
        self.rounds = 0
        
        self.aggregator = ModelAggregator()
        self.selector = ClientSelector()
        
        #SHAP utils
        self.shap_util = shap_util
        self.e = []
        self.shap_values = []
        self.shap_prediction = []
        
        # SHAP metrics
        self.positive_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.negative_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.positive_shap_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.negative_shap_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        self.non_zero_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        
        #Test utils
        self.criterion = F.nll_loss if self.config.MODELNAME == self.config.MNIST_NAME else nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        self.test_losses = []
        
        # raw performance measures
        self.confusion_matrix = torch.zeros(self.config.NUMBER_TARGETS, self.config.NUMBER_TARGETS)
        self.correct = 0
    
    def set_rounds(self, rounds):
        self.rounds = rounds
        
    def create_default_model(self):
        net = self.config.NETWORK()
        Path(os.path.dirname(self.default_model_path)).mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), self.default_model_path)
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
    
    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                output = self.net(data)
                out = output.log() if self.config.MODELNAME == self.config.MNIST_NAME else output
                test_loss += self.criterion(out, target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                for t, p in zip(target.view(-1), pred.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1
        test_loss /= len(self.test_dataloader.dataset)
        self.correct = correct
        print('\nServer Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_dataloader.dataset),
            100. * correct / len(self.test_dataloader.dataset)))
        self.test_losses.append(test_loss)
        
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
            self.positive_shap_mean[i] = [arr[(np.array(arr) > 0)].mean for arr in self.shap_values[i]]
            self.negative_shap_mean[i] = [arr[(np.array(arr) < 0)].mean for arr in self.shap_values[i]]
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
        self.observer.push_metrics(self.accuracy, self.recall, self.precision, self.positive_shap, self.negative_shap, self.non_zero_mean,self.positive_shap_mean, self.negative_shap_mean)
        