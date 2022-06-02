import random

def set_rounds(client_plane, server, rounds):
    client_plane.set_rounds(rounds)
    server.set_rounds(rounds)
    
def update_configs(client_plane, server, config, observer_config):
    client_plane.update_config(config, observer_config)
    server.update_config(config, observer_config)
    
def run_round(client_plane, server, rounds):
    # Federated Learning Round 
    set_rounds(client_plane, server, rounds)
    client_plane.update_clients(server.get_nn_parameters())
    selected_clients = server.select_clients()
    client_parameters = client_plane.train_selected_clients(selected_clients)
    server.aggregate_model(client_parameters)

def select_random_clean(client_plane, config, n):
    indices = []
    for i in range(n):
        idx = random.randint(0,config.NUMBER_OF_CLIENTS)
        while idx in client_plane.poisoned_clients or idx in indices:
            idx = random.randint(0,config.NUMBER_OF_CLIENTS)
        indices.append(idx)
    return indices

def select_poisoned(client_plane, n):
        return client_plane.poisoned_clients[:n]

def train_client(client_plane, rounds, idx): 
    client_plane.clients[idx].train(rounds)
    client_plane.clients[idx].push_metrics()

def print_posioned_target(client_plane, idx):
    client = client_plane.clients[idx]
    print(client.train_dataloader.dataset.dataset.targets[client.poisoned_indices][0])
    
    