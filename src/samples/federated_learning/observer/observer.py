from ..utils import VMUtil

class Observer(VMUtil):
    def __init__(self, config, observer_config):
        super(Observer, self).__init__(config)
        self.config = config
        self.observer_config = observer_config
        self.experiment_type = self.observer_config.experiment_type
        self.experiment_id = self.observer_config.experiment_id
        self.poisoned_clients = self.config.POISONED_CLIENTS
        self.test = self.observer_config.test
        self.dataset_type = self.observer_config.dataset_type