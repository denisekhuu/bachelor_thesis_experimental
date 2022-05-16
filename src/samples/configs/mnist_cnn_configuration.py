import os
import torch.nn as nn
from torch import device
from .nets import MNISTCNN, FashionMNISTCNN, MNISTSNN
from .dataset import MNISTDataset, FashionMNISTDataset
from .dataloader import MNISTDataloader, FashionMNISTDataloader
from .client import SNNClient, CNNClient, Client

class Configuration():
    
    # Dataset Config
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_TEST = 1000
    DATASET = MNISTDataset
    
    # DEPRICATED CONFIG
    DATALOADER = MNISTDataloader
    
    
    #MNIST_FASHION_DATASET Configurations
    MNIST_FASHION_DATASET_PATH = os.path.join('./data/mnist_fashion')
    MNIST_FASHION_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',  'Bag', 'Ankle Boot']
    
    #MNIST_DATASET Configurations
    MNIST_DATASET_PATH = os.path.join('./data/mnist')
    
    #CIFAR_DATASET Configurations
    CIFAR10_DATASET_PATH = os.path.join('./data/cifar10')
    CIFAR10_LABELS = ['Plane', 'Car', 'Bird', 'Cat','Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    #Model Training Configurations
    N_EPOCHS = 3
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    LOG_INTERVAL = 10
    CRITERION = nn.CrossEntropyLoss()
    NETWORK = MNISTCNN
    
    #Local Environment Configurations
    NUMBER_OF_CLIENTS = 1
    CLIENT_TYPE = CNNClient
    DEVICE = device('cpu')
    
    
    #Label Flipping Attack
    FROM_LABEL = 5
    TO_LABEL = 4