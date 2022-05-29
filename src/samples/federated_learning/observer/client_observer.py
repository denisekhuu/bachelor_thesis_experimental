from datetime import datetime
from .observer import Observer
import torch

class ClientObserver(Observer):
    def __init__(self, config, observer_config, client_id, poisoned, dataset_size):
        super(ClientObserver, self).__init__(config, observer_config)
        self.name = self.observer_config.client_name 
        self.client_id = client_id
        self.poisoned = poisoned
        self.poisoned_data = self.config.DATA_POISONING_PERCENTAGE
        self.num_epoch = self.config.N_EPOCHS
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.num_clients = self.config.NUMBER_OF_CLIENTS
        self.dataset_size = dataset_size
        self.type = self.observer_config.client_type
        self.metric_labels = { 
            "accuracy": "",
            "recall" : ",target={}",
            "precision" : ",target={}",
            "shap_pos": ",target={},source={}",
            "shap_neg": ",target={},source={}",
            "shap_mean": ",target={},source={}"
        }
        self.metrics = ["accuracy", "recall", "precision", "shap_pos", "shap_neg", "shap_mean"]
    
    def set_poisoned(poisoned):
        self.poisoned = poisoned
    
    def get_labels(self): 
        return "client_id={},test={},poisoned={},poisoned_data={},dataset_size={},type={},experiment_type={},experiment_id={},poisoned_clients={},num_of_epochs={},batch_size={},num_clients={}".format(
            self.client_id,
            self.test,
            self.poisoned,
            self.poisoned_data,
            self.dataset_size,
            self.type,
            self.experiment_type,
            self.experiment_id,
            self.poisoned_clients,
            self.num_epoch,
            self.batch_size,
            self.num_clients
        )
    
    def get_datastr(self, accuracy, recall, precision, shap_pos, shap_neg, shap_mean):
        timestamp = int(datetime.timestamp(datetime.now()))
        data = []
        labels = self.get_labels()
        datastr = "{},{} {} {}"
        data.append(datastr.format(self.name, labels, "accuracy=%f"%(accuracy), timestamp))
        for i in range(self.config.NUMBER_TARGETS): 
            data.append(datastr.format(self.name, labels + self.metric_labels["recall"].format(i), "recall=%f"%(recall[i]), timestamp))
            data.append(datastr.format(self.name, labels + self.metric_labels["precision"].format(i), "precision=%f"%(precision[i]), timestamp))
            for j in range(self.config.NUMBER_TARGETS): 
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_pos"].format(i, j), "shap_pos=%f"%(shap_pos[i][j]), timestamp))
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_neg"].format(i, j), "shap_neg=%f"%(shap_neg[i][j]), timestamp))
                data.append(datastr.format(self.name, labels + self.metric_labels["shap_mean"].format(i, j), "shap_mean=%f"%(shap_mean[i][j]), timestamp))
        return data
    
    def push_metrics(self, accuracy, recall, precision, shap_pos, shap_neg, shap_mean):
        data = self.get_datastr(accuracy, recall, precision, shap_pos, shap_neg, shap_mean)
        for d in data:
            self.push_data(d)
        print("Successfully pushed client data to victoria metrics")
        
        
        
                
                
        
        
    