import os
import torch.nn as nn
from torch import device
from .nets import MNISTCNN, FashionMNISTCNN
from .dataset import MNISTDataset, FashionMNISTDataset
from .dataloader import MNISTDataloader, FashionMNISTDataloader
from .client import FMNISTClient, MNISTClient

class Configuration():
    
    # Dataset Config
    BATCH_SIZE_TRAIN = 10
    BATCH_SIZE_TEST = 1000
    DATASET = MNISTDataset
    
    #MNIST_FASHION_DATASET Configurations
    MNIST_FASHION_DATASET_PATH = os.path.join('./data/mnist_fashion')
    MNIST_FASHION_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',  'Bag', 'Ankle Boot']
    
    #MNIST_DATASET Configurations
    MNIST_DATASET_PATH = os.path.join('./data/mnist')
    
    #CIFAR_DATASET Configurations
    CIFAR10_DATASET_PATH = os.path.join('./data/cifar10')
    CIFAR10_LABELS = ['Plane', 'Car', 'Bird', 'Cat','Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    #Model Training Configurations
    ROUNDS = 200
    N_EPOCHS = 1
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    LOG_INTERVAL = 200
    
    # Model Type Configurations
    MODELNAME = "MNISTCNN"
    NETWORK = MNISTCNN
    CLIENT_TYPE = MNISTClient
    NUMBER_TARGETS = 10
    
    #Local Environment Configurations
    NUMBER_OF_CLIENTS = 200
    DEVICE = device('cpu')
    
    #Label Flipping Attack 
    POISONED_CLIENTS = 50
    DATA_POISONING_PERCENTAGE = 1
    FROM_LABEL = 5
    TO_LABEL = 4
    
    #Victoria Metrics Configurations
    VM_URL = os.getenv('VM_URL') #URL settings in docker-compose.yml