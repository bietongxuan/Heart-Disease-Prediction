from collections import OrderedDict

import flwr as fl
from Model import *
import torch
from torch.nn import init


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Cnn().to(DEVICE)
for name, param in net.named_parameters():
    init.normal_(param, mean=0, std=0.01)
model_parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
acc=[]
class SaveModelStrategy(fl.server.strategy.FedAvg):
    # def aggregate_fit(self,server_round,results,failures,):
    #
    #     aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
    #
    #     if aggregated_parameters is not None:
    #         print(f"Saving round {server_round} aggregated_parameters...")
    #
    #         # Convert `Parameters` to `List[np.ndarray]`
    #         aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
    #
    #         # Convert `List[np.ndarray]` to PyTorch`state_dict`
    #         params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
    #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #         net.load_state_dict(state_dict, strict=True)
    #
    #         # Save the model
    #         torch.save(net.state_dict(), f"model_round_{server_round}.pth")
    #
    #     return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures, ):
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        acc.append(aggregated_accuracy)
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


#strategy = SaveModelStrategy(initial_parameters=fl.common.ndarrays_to_parameters(model_parameters))
strategy = SaveModelStrategy()
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=200), strategy=strategy)
print(max(acc))
# 模型导入
# model = Net()
# pre=torch.load(r'model_round_3.pth')
# model.load_state_dict(pre)
# print(model)
