import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import time

device = torch.device('cuda')
# Define the exact solution
def exact_solution_u(x, t):
    return (1.5*pi*torch.cos(x)+(x-pi*1.5))*torch.cos(t)

def initial_condition_u(x):
    return 1.5*pi*torch.cos(x) + (x - pi*1.5)

def initial_condition_u_t(x):
    return 0.0*pi*torch.cos(x)

# Define the exact solution
def exact_solution_p(x, t):
    return 1.5*pi*torch.sin(x)*torch.cos(t)

def initial_condition_p(x):
    return 1.5*pi*torch.sin(x)

def initial_condition_p_t(x):
    return 0.0*pi*torch.cos(x)

# assigning number of points
initial_pts = 500
left_boundary_pts = 500
right_boundary_pts = 500
residual_pts = 10000

# Type of optimizer (ADAM or LBFGS)
opt_type = "LBFGS"

eps = 5
manualSeed = 1

#np.random.seed(manualSeed)
#random.seed(manualSeed)
torch.manual_seed(manualSeed)

x_init = 3*pi*torch.rand((initial_pts,1)) # initial pts
t_init = 0*x_init
init =  torch.cat([x_init, t_init],1).to(device)
u_init = initial_condition_u(init[:,0]).reshape(-1, 1).to(device)
p_init = initial_condition_p(init[:,0]).reshape(-1, 1).to(device)
w_init = torch.cat([u_init, p_init],1).to(device)

u_t_init = initial_condition_u_t(init[:,0]).reshape(-1, 1).to(device)
p_t_init = initial_condition_p_t(init[:,0]).reshape(-1, 1).to(device)
w_t_init = torch.cat([u_t_init, p_t_init],1).to(device)



xb_left = torch.zeros((left_boundary_pts, 1)) # left spatial boundary
tb_left = torch.rand((left_boundary_pts, 1)) #
b_left = torch.cat([xb_left, tb_left ],1).to(device)
u_b_l = exact_solution_u(xb_left, tb_left).to(device)
p_b_l = exact_solution_p(xb_left, tb_left).to(device)
w_b_l = torch.cat([u_b_l, p_b_l],1).to(device)



xb_right = 3*pi*torch.ones((right_boundary_pts, 1)) # right spatial boundary
tb_right = torch.rand((right_boundary_pts, 1)) # right boundary pts
b_right = torch.cat([xb_right, tb_right ],1).to(device)
u_b_r = exact_solution_u(xb_right, tb_right).to(device)
p_b_r = exact_solution_p(xb_right, tb_right).to(device)
w_b_r = torch.cat([u_b_r, p_b_r],1).to(device)

# x_interior = pi*torch.rand((residual_pts, 1))
# t_interior = torch.rand((residual_pts, 1))
# interior = torch.cat([x_interior, t_interior],1)

# x_interior = torch.linspace(0, 3*pi, 102)
# #x_int = x_int[1:-1]

# t_interior = torch.linspace(0, 1, 102)
# #t_int = t_int[1:-1]

x_interior = 3*pi*torch.rand((residual_pts, 1))
t_interior = torch.rand((residual_pts, 1))
interior = torch.cat([x_interior, t_interior],1).to(device)

# torch.set_printoptions(threshold=10_000)


# n = 100  # size of matrix
# W = torch.tril(torch.ones(n, n), diagonal=-1)  # create a lower triangular matrix of ones
# W -= torch.diag(torch.diag(W))  # set the diagonal elements to zero




training_set = DataLoader(torch.utils.data.TensorDataset(init.to(device), w_init.to(device), w_t_init.to(device), b_left.to(device),  b_right.to(device)), batch_size=2000, shuffle=False)


class CustomActivation(nn.Module):
    def forward(self, x):
        return (1 - x**2) * torch.exp(-x**2)

class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = CustomActivation()

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

# Model definition
my_network = NeuralNet(input_dimension = init.shape[1], output_dimension = w_init.shape[1], n_hidden_layers=4, neurons=200)
my_network = my_network.to(device)

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            #torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)
    model.apply(init_weights)

# Random Seed for weight initialization
retrain = 128
# Xavier weight initialization
init_xavier(my_network, retrain)
#print(my_network(init))

if opt_type == "ADAM":
    optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
elif opt_type == "LBFGS":
    optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
else:
    raise ValueError("Optimizer not recognized")


def fit(model, training_set, interior, num_epochs, optimizer, p, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, (initial, w_initial, w_initial_t, bd_left, bd_right) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # for initial
                initial.requires_grad = True
                w_initial_pred_ = model(initial)
                u_initial_pred_ = w_initial_pred_[:, 0].reshape(-1, 1)
                p_initial_pred_ = w_initial_pred_[:, 1].reshape(-1, 1)

                # with derivative
                inpu = torch.ones(initial_pts, 1).to(device)

                grad_u_ini = \
                torch.autograd.grad(u_initial_pred_, initial, grad_outputs=inpu, create_graph=True, allow_unused=True)[
                    0]

                u_initial_t = grad_u_ini[:, 1]

                grad_p_ini = torch.autograd.grad(p_initial_pred_, initial, grad_outputs=inpu, create_graph=True)[0]

                p_initial_t = grad_p_ini[:, 1]

                # for left boundary
                w_bd_left_pred_ = model(bd_left)
                u_bd_left_pred_ = w_bd_left_pred_[:, 0].reshape(-1, 1)
                p_bd_left_pred_ = w_bd_left_pred_[:, 1].reshape(-1, 1)
                # for right boundary
                w_bd_right_pred_ = model(bd_right)
                u_bd_right_pred_ = w_bd_right_pred_[:, 0].reshape(-1, 1)
                p_bd_right_pred_ = w_bd_right_pred_[:, 1].reshape(-1, 1)

                #                 # residual calculation
                interior.requires_grad = True
                w_hat = model(interior)
                u_hat = w_hat[:, 0].reshape(-1, 1)
                p_hat = w_hat[:, 1].reshape(-1, 1)
                inputs = torch.ones(residual_pts, 1).to(device)
                #                 inputs2 = torch.ones(residual_pts, 1)
                grad_u_hat = \
                torch.autograd.grad(u_hat.reshape(-1, 1), interior, grad_outputs=inputs, create_graph=True)[0]
                u_x = grad_u_hat[:, 0].reshape(-1, 1)
                u_t = grad_u_hat[:, 1].reshape(-1, 1)

                grad_u_hat_xx = torch.autograd.grad(u_x, interior, grad_outputs=inputs, create_graph=True)[0]
                u_xx = grad_u_hat_xx[:, 0].reshape(-1, 1)

                u_tt = grad_u_hat_xx[:, 1].reshape(-1, 1)

                grad_p_hat = torch.autograd.grad(p_hat, interior, grad_outputs=inputs, create_graph=True)[0]

                p_x = grad_p_hat[:, 0].reshape(-1, 1)
                grad_p_hat_xx = torch.autograd.grad(p_x, interior, grad_outputs=inputs, create_graph=True)[0]
                p_xx = grad_p_hat_xx[:, 0].reshape(-1, 1)

                p_t = grad_p_hat[:, 1].reshape(-1, 1)
                grad_p_hat_tt = torch.autograd.grad(p_t, interior, grad_outputs=inputs, create_graph=True)[0]
                p_tt = grad_p_hat_tt[:, 1].reshape(-1, 1)

                loss_pde = (u_tt.reshape(-1, ) - u_xx.reshape(-1, ) + u_hat.reshape(-1, ) - p_x.reshape(-1, )) ** 2

                # #
                loss_pde1 = (p_tt.reshape(-1, ) + u_x.reshape(-1, ) - p_xx.reshape(-1, ) + p_hat.reshape(
                    -1, ) - torch.cos(interior[:, 1]).reshape(-1, )) ** 2

                #                 grad_grad_u_x = torch.autograd.grad(u_x, interior, grad_outputs=torch.ones(interior.shape[0]), create_graph=True)[0]
                #                 u_xx = grad_grad_u_x[:, 0]
                #                 u_t = grad_u_hat[:, 1].reshape(-1,1)
                #                 print(u_t)
                #                 grad_u_hat_tt = torch.autograd.grad(u_t, interior, grad_outputs=inputs, create_graph=True)[0]

                #                 u_tt = grad_u_hat_tt[:, 1].reshape(-1,1)

                #                 p_t = grad_p_hat[:,1].reshape(-1,1)
                #                 grad_p_hat_tt = torch.autograd.grad(p_t, interior, grad_outputs=inputs, create_graph=True)[0]

                #                 p_tt = grad_p_hat_tt[:, 1].reshape(-1,1)

                #                 pde_single_column = (u_tt.reshape(-1, ) - u_xx.reshape(-1, ) + u_hat.reshape(-1, ) - p_x.reshape(-1, )) ** 2
                #                 #pde_single_column = pde_single_column.reshape(-1, )

                #                 loss_pde = pde_single_column

                #                 pde_single_column1 = (p_tt.reshape(-1, )  + u_x.reshape(-1, ) - p_xx.reshape(-1, ) + p_hat.reshape(-1, ) - torch.cos(interior[:, 1]).reshape(-1, )) ** 2
                #                 #pde_single_column1 = pde_single_column1.reshape(-1, )

                #                 loss_pde1 = pde_single_column1

                # Item 1. below
                loss1 = torch.mean(loss_pde) + torch.mean(
                    (u_initial_t.reshape(-1, ) - w_initial_t[:, 0].reshape(-1, )) ** p) + torch.mean(
                    (u_initial_pred_.reshape(-1, ) - w_initial[:, 0].reshape(-1, )) ** p) + torch.mean(
                    (u_bd_left_pred_.reshape(-1, ) - u_b_l.reshape(-1, )) ** p) + torch.mean(
                    (u_bd_right_pred_.reshape(-1, ) - u_b_r.reshape(-1, )) ** p)
                loss2 = torch.mean(loss_pde1) + torch.mean(
                    (p_initial_pred_.reshape(-1, ) - w_initial[:, 1].reshape(-1, )) ** p) + torch.mean(
                    (p_initial_t.reshape(-1, ) - w_initial_t[:, 1].reshape(-1, )) ** p) + torch.mean(
                    (p_bd_left_pred_.reshape(-1, ) - p_b_l.reshape(-1, )) ** p) + torch.mean(
                    (p_bd_right_pred_.reshape(-1, ) - p_b_r.reshape(-1, )) ** p)
                loss = loss1 + loss2
                # loss = torch.max(torch.abs((u_initial_pred_.reshape(-1, ) - u_initial.reshape(-1, )))) + torch.max(torch.abs((u_t.reshape(-1, ) - u_xx.reshape(-1, ))))+torch.max(torch.abs((u_bd_left_pred_.reshape(-1,)))) + torch.max(torch.abs((u_bd_right_pred_.reshape(-1,))))

                # Item 2. below
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item()
                return loss

            # Item 3. below
            optimizer.step(closure=closure)

        print('Loss: ', (running_loss[0] / len(training_set)))
        history.append(running_loss[0])

    return history


start_time = time.time()
n_epochs = 10000
history = fit(my_network, training_set, interior, n_epochs, optimizer_, p=2, verbose=True )
end_time = time.time()
total_time = end_time - start_time

# print("Training time: {:.2f} seconds".format(total_time))

# import pickle

# # Load the history from the pickle file
# with open('PINN_tim_mexican_hat.pkl', 'rb') as f:
#     history = pickle.load(f)

# # # Load the model architecture
# # my_network = your_model_module.YourModelClass()  # Instantiate your model class

# # Load the saved model state dictionary
# # Load the saved model state dictionary
# model_state_dict = torch.load('PINN_tim_mexican_hat.pth', map_location=torch.device('cpu')) 

# # # with open('PINN_tim_mexican_hat.pkl', 'wb') as f:
# #     pickle.dump(history, f)

# f.close()

# model_state_dict = my_network.state_dict()

# # Save the model state dictionary to a file
# torch.save(model_state_dict, 'PINN_tim_mexican_hat.pth')

x_test = torch.linspace(0, 3*pi, 10000).reshape(-1,1)
t_test = torch.ones((10000,1))
test = torch.cat([x_test, t_test],1)
u_test = exact_solution_u(x_test, t_test).reshape(-1,1)
p_test = exact_solution_p(x_test, t_test).reshape(-1,1)
my_network = my_network.cpu()
w_test_pred = my_network(test)
u_test_pred = w_test_pred[:,0].reshape(-1,1)
p_test_pred = w_test_pred[:,1].reshape(-1,1)

# plt.grid(True, which="both", ls=":")
# plt.plot(x_test, u_test, label="Ground Truth",lw=2)
# plt.plot(x_test, u_test_pred.detach(), label="Network Prediction",lw=2)
# plt.xlabel("x")
# plt.ylabel("u")
# plt.legend()

# Compute the relative L2 error norm (generalization error)
relative_error_test = torch.mean((u_test_pred - u_test)**2)/torch.mean(u_test**2)
#relative_error_test = torch.max(torch.abs(u_test_pred -u_test))/torch.max(torch.abs(u_test))
print("Relative Error Test: ", relative_error_test.detach().numpy()*100, "%")


# Compute the relative L2 error norm (generalization error)
relative_error_test = torch.mean((p_test_pred - p_test)**2)/torch.mean(p_test**2)
#relative_error_test = torch.max(torch.abs(u_test_pred -u_test))/torch.max(torch.abs(u_test))
print("Relative Error Test: ", relative_error_test.detach().numpy()*100, "%")


import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Assuming you have imported your data and defined necessary functions

# Rest of your code...

# # Convert the font size to points
font_size = 20
# ticks_font = FontProperties(family='Times New Roman', style='normal', size=font_size)




x_test = 3*pi*torch.rand(100000).reshape(-1,1)
t_test = torch.rand(100000).reshape(-1,1)
test = torch.cat([x_test, t_test],1)
u_test = exact_solution_u(x_test, t_test).reshape(-1,1)
p_test = exact_solution_p(x_test, t_test).reshape(-1,1)
my_network = my_network.cpu()
w_test_pred = my_network(test)
u_test_pred = w_test_pred[:,0].reshape(-1,1)
p_test_pred = w_test_pred[:,1].reshape(-1,1)


# # reshaping and detach numpy
x_test = x_test.reshape(-1, )
t_test = t_test.reshape(-1, )

u_test_pred = u_test_pred.reshape(-1, )

x_test = x_test.detach().numpy()
t_test = t_test.detach().numpy()
u_test_pred = u_test_pred.detach().numpy()
# relative_error = relative_error.detach().numpy()





plt.rcParams['figure.figsize'] = [10, 4]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False
font_path = 'times-new-roman.ttf'
custom_font = FontProperties(fname=font_path)

ticks_font = FontProperties(fname=font_path)

# Define the levels for contouring
levels = np.linspace(-8, 8, 20)

        
#CS1 = plt.tricontourf(x_test, t_test, u_test_pred, 20, cmap='twilight')
CS1 = plt.tricontourf(x_test, t_test, u_test_pred, levels, cmap='Spectral_r')
#CS1 = plt.tricontourf(x_test, t_test, relative_error, 20, cmap='rainbow')



cbar1 = plt.colorbar(CS1)
for t in cbar1.ax.get_yticklabels():
    t.set_fontproperties(custom_font)
    t.set_fontsize(12)



plt.xlabel('x', fontsize=20, fontproperties=custom_font)
plt.ylabel('t', fontsize=20, fontproperties=custom_font)
plt.xticks(fontsize=20, fontproperties=custom_font)
plt.yticks(fontsize=20, fontproperties=custom_font)
#plt.savefig('Causal_EB.pdf', dpi = 300, bbox_inches = "tight")
#plt.savefig('Exact_EB.pdf', dpi = 500, bbox_inches = "tight", format='pdf', backend='cairo')
#plt.savefig('Absolute_error_EB.pdf', dpi = 300, bbox_inches = "tight")


# # Plotting
# plt.rcParams['figure.figsize'] = [10, 4]
# plt.rcParams['axes.unicode_minus'] = False

# # plt.tricontourf(x_test, t_test, u_test, 20, cmap='rainbow')
# # cbar1 = plt.colorbar()
# # cbar1.ax.tick_params(labelsize=font_size)
# # for t in cbar1.ax.get_yticklabels():
# #     t.set_fontproperties(ticks_font)

# plt.xlabel('x', fontsize=font_size, fontproperties=ticks_font)
# plt.ylabel('t', fontsize=font_size, fontproperties=ticks_font)
# plt.xticks(fontsize=font_size, fontproperties=ticks_font)
# plt.yticks(fontsize=font_size, fontproperties=ticks_font)

# plt.savefig('PINN_timo_disp_adaptive_new.pdf', dpi=500, bbox_inches="tight", format='pdf')
plt.savefig('PINN_timo_disp_wavelet_new.png', dpi=500, bbox_inches="tight", format='png')
#plt.savefig('Causal_EB.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Assuming you have imported your data and defined necessary functions

# Rest of your code...

# # Convert the font size to points
font_size = 20
# ticks_font = FontProperties(family='Times New Roman', style='normal', size=font_size)




x_test = 3*pi*torch.rand(100000).reshape(-1,1)
t_test = torch.rand(100000).reshape(-1,1)
test = torch.cat([x_test, t_test],1)
u_test = exact_solution_u(x_test, t_test).reshape(-1,1)
p_test = exact_solution_p(x_test, t_test).reshape(-1,1)
my_network = my_network.cpu()
w_test_pred = my_network(test)
u_test_pred = w_test_pred[:,0].reshape(-1,1)
p_test_pred = w_test_pred[:,1].reshape(-1,1)


# # reshaping and detach numpy
x_test = x_test.reshape(-1, )
t_test = t_test.reshape(-1, )

p_test_pred = p_test_pred.reshape(-1, )

x_test = x_test.detach().numpy()
t_test = t_test.detach().numpy()
p_test_pred = p_test_pred.detach().numpy()
# relative_error = relative_error.detach().numpy()





plt.rcParams['figure.figsize'] = [10, 4]

from matplotlib.font_manager import FontProperties
plt.rcParams['axes.unicode_minus'] = False
font_path = 'times-new-roman.ttf'
custom_font = FontProperties(fname=font_path)

ticks_font = FontProperties(fname=font_path)

# Define the levels for contouring
levels = np.linspace(-6, 6, 20)

        
#CS1 = plt.tricontourf(x_test, t_test, u_test_pred, 20, cmap='twilight')
CS1 = plt.tricontourf(x_test, t_test, p_test_pred, levels, cmap='Spectral_r')
#CS1 = plt.tricontourf(x_test, t_test, relative_error, 20, cmap='rainbow')



#cbar1 = plt.colorbar(CS1)
for t in cbar1.ax.get_yticklabels():
    t.set_fontproperties(custom_font)
    t.set_fontsize(12)



plt.xlabel('x', fontsize=20, fontproperties=custom_font)
plt.ylabel('t', fontsize=20, fontproperties=custom_font)
plt.xticks(fontsize=20, fontproperties=custom_font)
plt.yticks(fontsize=20, fontproperties=custom_font)
#plt.savefig('Causal_EB.pdf', dpi = 300, bbox_inches = "tight")
#plt.savefig('Exact_EB.pdf', dpi = 500, bbox_inches = "tight", format='pdf', backend='cairo')
#plt.savefig('Absolute_error_EB.pdf', dpi = 300, bbox_inches = "tight")


# # Plotting
# plt.rcParams['figure.figsize'] = [10, 4]
# plt.rcParams['axes.unicode_minus'] = False

# # plt.tricontourf(x_test, t_test, u_test, 20, cmap='rainbow')
# # cbar1 = plt.colorbar()
# # cbar1.ax.tick_params(labelsize=font_size)
# # for t in cbar1.ax.get_yticklabels():
# #     t.set_fontproperties(ticks_font)

# plt.xlabel('x', fontsize=font_size, fontproperties=ticks_font)
# plt.ylabel('t', fontsize=font_size, fontproperties=ticks_font)
# plt.xticks(fontsize=font_size, fontproperties=ticks_font)
# plt.yticks(fontsize=font_size, fontproperties=ticks_font)
plt.savefig('PINN_timo_rotation_wavelet_new.png', dpi=500, bbox_inches="tight", format='png')

# plt.savefig('PINN_timo_rotation_adaptive_new.pdf', dpi=500, bbox_inches="tight", format='pdf')
#plt.savefig('Causal_EB.pdf', dpi=500, bbox_inches="tight", format='pdf', backend='cairo')


