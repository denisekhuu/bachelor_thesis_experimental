import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from .client import Client

class FFNNClient(Client): 
    
    def __init__(self, configs, train_dataloader, test_dataloader, shap_util):
        super(FFNNClient, self).__init__(configs, train_dataloader, test_dataloader, shap_util)
        self.criterion = F.nll_loss
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.5)
        
    def train(self, epoch):
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.configs.DEVICE), target.to(self.configs.DEVICE)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output.log(), target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append((batch_idx*64) + ((epoch-1)*len(self.train_dataloader.dataset)))

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.configs.DEVICE), target.to(self.configs.DEVICE)
                output = self.net(data)
                test_loss += self.criterion(output.log(), target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                for t, p in zip(target.view(-1), pred.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1
        test_loss /= len(self.test_dataloader.dataset)
        self.correct = correct
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_dataloader.dataset),
            100. * correct / len(self.test_dataloader.dataset)))
        self.test_losses.append(test_loss)
        