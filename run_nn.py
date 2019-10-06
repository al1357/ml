import neural_network as nn
import data_loader


dl = data_loader.data_loader([0.3, 0])
dl.load_iris()

network_structure = [24, 3]

nn1 = nn.neural_network(dl, \
                        network_map = network_structure, \
                        load_parameters=False)

nn1.learn(gradient_check=True)
local_nn1 = nn1.__dict__


