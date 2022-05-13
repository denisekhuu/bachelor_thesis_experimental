import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

from .client import Client

class CNNClient(Client): 
    
    def __init__(self, configs, train_dataloader, test_dataloader):
        super(CNNClient, self).__init__(configs, train_dataloader, test_dataloader)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.configs.LEARNING_RATE, momentum=self.configs.MOMENTUM)
        self.criterion = nn.CrossEntropyLoss()

        
    def train(self, epoch):
        """
        Defines the training process of a local model. train() uses as optimization algorithm Stochastic 
        Gradient Descent (SGD) and Cross Entropy Loss as loss-function.
        :param epoch: epoch
        :type epoch: int
        """
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.configs.DEVICE), target.to(self.configs.DEVICE)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.configs.LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(self.train_dataloader.dataset),100. * batch_idx / len(self.train_dataloader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_counter.append((batch_idx*64) + ((epoch-1)*len(self.train_dataloader.dataset)))
                #self.torch.save(self.net.state_dict(), './results/model.pth')
                #torch.save(optimizer.state_dict(), './results/optimizer.pth')
    
    def test(self):
        """
        Test function to evaluate the performance of the deep learning model
        """
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.configs.DEVICE), target.to(self.configs.DEVICE)
                output = self.net(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                total += 1
        test_loss /= total
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_dataloader.dataset), 100. * correct / len(self.test_dataloader.dataset)))