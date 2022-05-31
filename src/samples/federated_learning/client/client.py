import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
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
        
        # training config
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        self.criterion = F.nll_loss if self.config.MODELNAME == self.config.MNIST_NAME else nn.CrossEntropyLoss()
        
        # datasets
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        # loss metrics
        #self.train_losses = []
        #self.train_counter = []
        #self.test_losses = []
        #self.test_counter = [i*len(self.train_dataloader.dataset) for i in range(self.config.N_EPOCHS)]
        
        # raw performance measures
        self.confusion_matrix = torch.zeros(self.config.NUMBER_TARGETS, self.config.NUMBER_TARGETS)
        self.correct = 0
        
        # SHAP utils
        self.e = None
        
        # label flipping meta data
        self.is_poisoned = False
        self.poisoned_indices = []
        self.poisoning_indices = []
        
        self.load_default_model()
        
    def train(self, epoch):
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            self.optimizer.zero_grad()
            output = self.net(data)
            out = output.log() if self.config.MODELNAME == self.config.MNIST_NAME else output
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % self.config.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    (epoch+1), batch_idx * len(data), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))
                #self.train_losses.append(loss.item())
                #self.train_counter.append((batch_idx*64) + ((epoch-1)*len(self.train_dataloader.dataset)))
        
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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_dataloader.dataset),
            100. * correct / len(self.test_dataloader.dataset)))
        #self.test_losses.append(test_loss)
        
        
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
        return self.shap_util.get_shap_values(self.e)
    
    def get_shap_predictions(self):
        return self.shap_util.predict(self.net)
        
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
        recall = self.confusion_matrix.diag()/self.confusion_matrix.sum(1)
        precision = self.confusion_matrix.diag()/self.confusion_matrix.sum(0)
        accuracy = self.correct / len(self.test_dataloader.dataset)
        precision[torch.isnan(precision)] = 0
        recall[torch.isnan(recall)] = 0
        return recall, precision, accuracy
        
        
    def analize_shap_values(self, shap_values): 
        """
        Calculate SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        """
        positive_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        negative_shap = [[] for i in range(self.config.NUMBER_TARGETS)]
        positive_shap_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        negative_shap_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        non_zero_mean = [[] for i in range(self.config.NUMBER_TARGETS)]
        for i in range(self.config.NUMBER_TARGETS):
            positive_shap[i] = [np.sum(np.array(arr) > 0) for arr in shap_values[i]]
            negative_shap[i] = [np.sum(np.array(arr) < 0) for arr in shap_values[i]]
            positive_shap_mean[i] = [arr[np.array(arr) > 0].mean() for arr in shap_values[i]]
            negative_shap_mean[i] = [arr[np.array(arr) < 0].mean() for arr in shap_values[i]]
            non_zero_mean[i] = [arr[np.nonzero(arr)].mean() for arr in shap_values[i]]
            
        
        return positive_shap, negative_shap, non_zero_mean, positive_shap_mean, negative_shap_mean

    def analize(self):
        """
        Calculate SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        and test metrics like accuracy, precision and recall
        """
        recall, precision, accuracy = self.analize_test()
        shap_values = self.get_shap_values()
        positive_shap, negative_shap, non_zero_mean, positive_shap_mean, negative_shap_mean = self.analize_shap_values(shap_values)
        return recall, precision, accuracy, positive_shap, negative_shap, non_zero_mean, positive_shap_mean, negative_shap_mean
    
    def push_metrics(self, timestamp=None): 
        """
        Push SHAP metrics like number of positive and negativ SHAP values as well as non-zero-mean
        and test metrics like accuracy, precision and recall to victoria metrics
        """
        recall, precision, accuracy, positive_shap, negative_shap, non_zero_mean, positive_shap_mean, negative_shap_mean = self.analize()
        self.observer.push_metrics(recall, precision, accuracy, positive_shap, negative_shap, non_zero_mean, positive_shap_mean, negative_shap_mean, timestamp)
        
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
        self.confusion_matrix = torch.zeros(self.config.NUMBER_TARGETS, self.config.NUMBER_TARGETS)
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
        self.net.eval()
        
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