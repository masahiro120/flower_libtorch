import flwr as fl
import numpy as np
from fedavg_cpp import FedAvgCpp, weights_to_parameters

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    initial_weights = []
    initial_parameters = weights_to_parameters(initial_weights)
    strategy = FedAvgCpp(initial_parameters=initial_parameters)
    fl.server.start_server(
        server_address="0.0.0.0:8888",
        config={"num_rounds": 10},
        strategy=strategy,
    )
