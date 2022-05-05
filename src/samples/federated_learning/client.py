import torch
import torch.optim as optim
import torch.nn as nn
from .configuration import Configuration

class Client(): 
    def __init__(self, configs: Configuration, train_dataloader, test_dataloader):
        self.configs = configs
        self.net = self.configs.NETWORK()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.configs.LEARNING_RATE, momentum=self.configs.MOMENTUM)
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(self.train_dataloader.dataset) for i in range(self.configs.N_EPOCHS)]
        
    def train(self, epoch):
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
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
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                output = self.net(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                total += 1
        test_loss /= total
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_dataloader.dataset), 100. * correct / len(self.test_dataloader.dataset)))