class ModelAggregator():
    def model_avg(self, parameters):
        new_params = {}
        for name in parameters[0].keys():
            new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)
        return new_params