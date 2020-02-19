import neural_network as nn
import data_loader


dl = data_loader.data_loader([0.3, 0])
#dl.load_mnist(less_than=4)
#dl.load_iris()
dl.load_cars()

# =============================================================================
# x = dl.get_x()
# y = dl.get_y()
# 
# cv_x = dl.get_x(kind="cv")
# cv_y = dl.get_y(kind="cv")
# =============================================================================

network_structure = [20, 1]

nn1 = nn.neural_network(dl, \
                        network_map = network_structure, \
                        load_parameters=False)

nn1.learn(gradient_check=False)
local_nn1 = nn1.__dict__
