import shap
import numpy as np
import torch
import os

class SHAPUtil():
    def __init__(self, data_loader, net):
        """
        Simulation of isolated distributed clients
        :param data_loader: data to extract shap images and shap labels
        :type data_loader: torch.utils.data.DataLoader
        :param clients: distributed clients
        :type clients: Client[]
        """
        self.data_loader = data_loader
        self.net = net
        self.images, self.targets = self.get_SHAP_dataset()
        self.shap_images, self.shap_indices = self.get_SHAP_sample()
        self.background = self.images[:100]
        self.e, self.shap_values = self.get_shap_values()
        self.prediction = self.predict()
        
    def get_SHAP_dataset(self):
        """
        Load test sample
        return Tensor, Tensor
        """
        examples = enumerate(self.data_loader)
        batch_idx, (images, targets) = next(examples)
        return images, targets
        
    def get_SHAP_sample(self):
        """
        Get last occurence of a label in target sample
        return int[]
        """
        indices=[]
        for i in range(10):
            try:
                index = torch.where(self.targets==i)
                indices.append(index[0][-1].item())
            except: 
                print("does not exist")
        return self.images[indices], indices
                    
    def get_shap_values(self):
        """
        Calculate DeepExplainer and SHAP values based on sample
        return shap.DeepExplainer, array
        """
        e = shap.DeepExplainer(self.net, self.background)
        shap_values = e.shap_values(self.shap_images)
        return e, shap_values

    
    def plot_shap_images(self):
        """
        Plot sample images and their target labels
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i, idx in enumerate(self.shap_indices):
            plt.subplot(3,4,i+1)
            plt.tight_layout()
            plt.imshow(self.images[idx][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(self.targets[idx]))
            plt.xticks([])
            plt.yticks([])
        plt.show()
        
    def predict(self):
        """
        Predict SHAP test images
        return Tensor
        """
        output = self.net(self.images[self.shap_indices])
        pred = output.data.max(1, keepdim=True)[1]
        print("Predictions": pred)
        return pred
    
    def plot(self, file):
        """
        Plot SHAP values and image
        :param file: name of file
        :type clients: os.path
        """
        import matplotlib.pyplot as plt
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in self.shap_values]
        test_numpy = np.swapaxes(np.swapaxes(self.shap_images.numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy, show=False)
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        plt.savefig(file)
        
        
    def analize(self):
        pass
    