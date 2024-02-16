import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pickle
import time

device = torch.device('cuda')

wt_1 = torch.zeros(10000, 1)
wt_2 = torch.zeros(10000, 1)
wt_3 = torch.zeros(10000, 1)
wt_4 = torch.zeros(10000, 1)
wt_5 = torch.zeros(10000, 1)
#wt_6 = torch.zeros(10000, 1)


# loss_1 = torch.zeros(10000, 1)
# loss_2 = torch.zeros(10000, 1)
# loss_3 = torch.zeros(10000, 1)
# loss_4 = torch.zeros(10000, 1)
# loss_5 = torch.zeros(10000, 1)
# loss_6 = torch.zeros(10000, 1)

error_1 = torch.zeros(10000, 1)
error_2 = torch.zeros(10000, 1)
error_3 = torch.zeros(10000, 1)
error_4 = torch.zeros(10000, 1)
error_5 = torch.zeros(10000, 1)
#error_6 = torch.zeros(10000, 1)

# Causality param
eps = 5

# Define the exact solution
def exact_solution(x, t):
    return torch.sin(x)*torch.cos(pi*t)

def initial_condition(x):
    return torch.sin(x)

def initial_condition_t(x):
    return 0*torch.cos(x)

# assigning number of points
initial_pts = 500
left_boundary_pts = 500
right_boundary_pts = 500
residual_pts = 10000

# Type of optimizer (ADAM or LBFGS)
opt_type = "LBFGS"

x_init = 8*pi*torch.rand((initial_pts,1)) # initial pts
t_init = 0*x_init
init = torch.cat([x_init, t_init],1).to(device)
u_init = initial_condition(init[:,0]).reshape(-1, 1)
u_init_t = 0*initial_condition(init[:,0]).reshape(-1, 1)

xb_left = torch.zeros((left_boundary_pts, 1)) # left spatial boundary
tb_left = torch.rand((left_boundary_pts, 1)) #
b_left = torch.cat([xb_left, tb_left ],1).to(device)
u_b_l = 0*torch.sin(tb_left)

xb_right = 8*pi*torch.ones((right_boundary_pts, 1)) # right spatial boundary
tb_right = torch.rand((right_boundary_pts, 1)) # right boundary pts
b_right = torch.cat([xb_right, tb_right ],1).to(device)
u_b_r = 0*torch.sin(2*pi - tb_right)

x_int = torch.linspace(0, 8*pi, 102)
x_int = x_int[1:-1]

t_int = torch.linspace(0, 1, 102)
t_int = t_int[1:-1]

x_interior = x_int.tile((100,))
x_interior = x_interior.reshape(-1,1)

t_interior = t_int.repeat_interleave(100)
t_interior = t_interior.reshape(-1,1)

# torch.set_printoptions(threshold=10_000)

interior = torch.cat([x_interior, t_interior],1).to(device)

n = 100  # size of matrix
W = torch.tril(torch.ones(n, n), diagonal=-1).to(device)  # create a lower triangular matrix of ones
W -= torch.diag(torch.diag(W)).to(device)  # set the diagonal elements to zero

training_set = DataLoader(torch.utils.data.TensorDataset(init.to(device), u_init.to(device), u_init_t.to(device), b_left.to(device),  b_right.to(device), u_b_l.to(device), u_b_r.to(device)), batch_size=500, shuffle=False)

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
        self.activation = nn.Tanh()

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
my_network = NeuralNet(input_dimension = init.shape[1], output_dimension = u_init.shape[1], n_hidden_layers=4, neurons=200)
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
        for j, (initial, u_initial, u_initial_t, bd_left, bd_right, ubl, ubr) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # for initial
                initial.requires_grad = True
                u_initial_pred_ = model(initial)
                inputs = torch.ones(initial_pts, 1).to(device)
                grad_u_init = torch.autograd.grad(u_initial_pred_, initial, grad_outputs=inputs, create_graph=True)[0]
                u_init_t = grad_u_init[:, 1].reshape(-1, )

                # for left boundary
                bd_left.requires_grad = True
                bd_left_pred_ = model(bd_left)
                inputs = torch.ones(left_boundary_pts, 1).to(device)
                grad_bd_left = torch.autograd.grad(bd_left_pred_, bd_left, grad_outputs=inputs, create_graph=True)[0]
                u_bd_x_left = grad_bd_left[:, 0].reshape(-1, )
                inputs = torch.ones(left_boundary_pts, 1).reshape(-1, ).to(device)
                grad_u_bd_x_left = torch.autograd.grad(u_bd_x_left, bd_left, grad_outputs=inputs, create_graph=True)[0]
                u_bd_xx_left = grad_u_bd_x_left[:, 0].reshape(-1, )
                #inputs = torch.ones(left_boundary_pts, 1).reshape(-1, )
                #grad_u_bd_xx_left = torch.autograd.grad(u_bd_xx_left, bd_left, grad_outputs=inputs, create_graph=True)[0]
                #u_bd_xxx_left = grad_u_bd_xx_left[:, 0].reshape(-1, )

                # for right boundary
                bd_right.requires_grad = True
                bd_right_pred_ = model(bd_right)
                inputs = torch.ones(right_boundary_pts, 1).to(device)
                grad_bd_right = torch.autograd.grad(bd_right_pred_, bd_right, grad_outputs=inputs, create_graph=True)[0]
                u_bd_x_right = grad_bd_right[:, 0].reshape(-1, )
                inputs = torch.ones(right_boundary_pts, 1).reshape(-1, ).to(device)
                grad_u_bd_x_right = torch.autograd.grad(u_bd_x_right, bd_right, grad_outputs=inputs, create_graph=True)[0]
                u_bd_xx_right = grad_u_bd_x_right[:, 0].reshape(-1, )

                # residual calculation
                interior.requires_grad = True
                u_hat = model(interior)
                inputs = torch.ones(residual_pts, 1).to(device)
                grad_u_hat = torch.autograd.grad(u_hat, interior, grad_outputs=inputs, create_graph=True)[0]

                u_x = grad_u_hat[:, 0].reshape(-1, )
                inputs = torch.ones(residual_pts, 1).reshape(-1, ).to(device)
                grad_u_x = torch.autograd.grad(u_x, interior, grad_outputs=inputs, create_graph=True)[0]
                u_xx = grad_u_x[:, 0].reshape(-1, )
                inputs = torch.ones(residual_pts, 1).reshape(-1, ).to(device)
                grad_u_xx = torch.autograd.grad(u_xx, interior, grad_outputs=inputs, create_graph=True)[0]
                u_xxx = grad_u_xx[:, 0].reshape(-1, )
                inputs = torch.ones(residual_pts, 1).reshape(-1, ).to(device)
                grad_u_xxx = torch.autograd.grad(u_xxx, interior, grad_outputs=inputs, create_graph=True)[0]
                u_xxxx = grad_u_xxx[:, 0].reshape(-1, )

                u_t = grad_u_hat[:, 1]
                inputs = torch.ones(residual_pts, 1).reshape(-1, ).to(device)
                grad_u_t = torch.autograd.grad(u_t, interior, grad_outputs=inputs, create_graph=True)[0]
                u_tt = grad_u_t[:, 1].reshape(-1, )

                pde_single_column = (u_tt.reshape(-1, ) + u_xxxx.reshape(-1, ) + \
                                     u_hat.reshape(-1, ) - (2-pi**2)*torch.sin(interior[:,0])*torch.cos(pi*interior[:,1])) ** 2
                pde_single_column = pde_single_column.reshape(-1, 1)

                pde_matrix = pde_single_column.reshape(100, 100)

                loss_at_time_steps = torch.mean(pde_matrix, 1)
                loss_at_time_steps = loss_at_time_steps.reshape(-1, 1)

                with torch.no_grad():
                    weighted_loss = torch.matmul(W, loss_at_time_steps).to(device)
                weighted_loss = torch.exp(-eps * weighted_loss)

                loss_pde = torch.mean(weighted_loss * loss_at_time_steps)
              
                weight1 = weighted_loss[10]
                weight2 = weighted_loss[20]
                weight3 = weighted_loss[30]
                weight4 = weighted_loss[40]
                weight5 = weighted_loss[50]
              
                #weight6 = weighted_loss[99]
                
#                 print("w1 {:.32f}".format(weight1.item()))

#                 print("w2 {:.32f}".format(weight2.item()))
#                 print("w3 {:.32f}".format(weight3.item()))
#                 print("w4 {:.32f}".format(weight4.item()))
#                 print("w5 {:.32f}".format(weight5.item()))
#                 print("w6 {:.32f}".format(weight6.item()))
                
                wt_1[epoch, :] = weight1
                wt_2[epoch, :] = weight2
                wt_3[epoch, :] = weight3
                wt_4[epoch, :] = weight4
                wt_5[epoch, :] = weight5
                #wt_6[epoch, :] = weight6
                
                
                
                
                
#                 loss1 = loss_at_time_steps[0]
#                 loss2 = loss_at_time_steps[19]
#                 loss3 = loss_at_time_steps[39]
#                 loss4 = loss_at_time_steps[59]
#                 loss5 = loss_at_time_steps[79]
#                 loss6 = loss_at_time_steps[99]
                
# #                 print("w1 {:.32f}".format(weight1.item()))

# #                 print("w2 {:.32f}".format(weight2.item()))
# #                 print("w3 {:.32f}".format(weight3.item()))
# #                 print("w4 {:.32f}".format(weight4.item()))
# #                 print("w5 {:.32f}".format(weight5.item()))
# #                 print("w6 {:.32f}".format(weight6.item()))
                
#                 loss_1[epoch, :] = loss1
#                 loss_2[epoch, :] = loss2
#                 loss_3[epoch, :] = loss3
#                 loss_4[epoch, :] = loss4
#                 loss_5[epoch, :] = loss5
#                 loss_6[epoch, :] = loss6
            
            

                # Item 1. below

                loss_ic = torch.mean((u_initial_pred_.reshape(-1, ) - u_initial.reshape(-1, )) ** p) + \
                          torch.mean((u_init_t.reshape(-1, )) ** p)
                #loss_pde = torch.mean((u_tt.reshape(-1, ) + u_xxxx.reshape(-1, )) ** p)
                loss_left_b = torch.mean((bd_left_pred_.reshape(-1, )) ** p) + \
                              torch.mean((u_bd_xx_left.reshape(-1, )) ** p)
                loss_right_b = torch.mean((bd_right_pred_.reshape(-1, )) ** p) + \
                               torch.mean((u_bd_xx_right.reshape(-1, )) ** p)

                loss = loss_ic + loss_pde + loss_left_b + loss_right_b
                
                
                # Test the model without accumulating gradients
                with torch.no_grad():
                    
                    # Set the model to evaluation mode (e.g., disables dropout)
                    my_network.eval()
                    
                    # Test data
                    x_test1 = torch.linspace(0, 8*pi, 100).reshape(-1,1)
                    t_test1 = 0.10*torch.ones((100,1))
                    test1 = torch.cat([x_test1, t_test1],1).to(device)
                    u_test1 = exact_solution(x_test1, t_test1).reshape(-1,1)
                    #my_network = my_network.to(device)
                    u_test_pred1 = my_network(test1).reshape(-1,1).to('cpu')
                    
                    # Compute the relative L2 error norm (generalization error)
                    error_1[epoch, :] = torch.mean((u_test_pred1 - u_test1)**2)/torch.mean(u_test1**2)
                    
                    
                    # Test data
                    t_test2 = 0.20*torch.ones((100,1))
                    test2 = torch.cat([x_test1, t_test2],1).to(device)
                    u_test2 = exact_solution(x_test1, t_test2).reshape(-1,1)
                    #my_network = my_network.to(device)
                    u_test_pred2 = my_network(test2).reshape(-1,1).to('cpu')
                    
                    # Compute the relative L2 error norm (generalization error)
                    error_2[epoch, :] = torch.mean((u_test_pred2 - u_test2)**2)/torch.mean(u_test2**2)
                    
                    
                    # Test data
                    t_test3 = 0.30*torch.ones((100,1))
                    test3 = torch.cat([x_test1, t_test3],1).to(device)
                    u_test3 = exact_solution(x_test1, t_test3).reshape(-1,1)
                    #my_network = my_network.to(device)
                    u_test_pred3 = my_network(test3).reshape(-1,1).to('cpu')
                    
                    # Compute the relative L2 error norm (generalization error)
                    error_3[epoch, :] = torch.mean((u_test_pred3 - u_test3)**2)/torch.mean(u_test3**2)
                    
                    
                    # Test data
                    t_test4 = 0.40*torch.ones((100,1))
                    test4 = torch.cat([x_test1, t_test4],1).to(device)
                    u_test4 = exact_solution(x_test1, t_test4).reshape(-1,1)
                    #my_network = my_network.to(device)
                    u_test_pred4 = my_network(test4).reshape(-1,1).to('cpu')
                    
                    # Compute the relative L2 error norm (generalization error)
                    error_4[epoch, :] = torch.mean((u_test_pred4 - u_test4)**2)/torch.mean(u_test4**2)
                    
                    
                    # Test data
                    t_test5 = 0.50*torch.ones((100,1))
                    test5 = torch.cat([x_test1, t_test5],1).to(device)
                    u_test5 = exact_solution(x_test1, t_test5).reshape(-1,1)
                    #my_network = my_network.to(device)
                    u_test_pred5 = my_network(test5).reshape(-1,1).to('cpu')
                    
                    # Compute the relative L2 error norm (generalization error)
                    error_5[epoch, :] = torch.mean((u_test_pred5 - u_test5)**2)/torch.mean(u_test5**2)
                    
                    
#                     # Test data
#                     t_test6 = 0.99*torch.ones((100,1))
#                     test6 = torch.cat([x_test1, t_test6],1).to(device)
#                     u_test6 = exact_solution(x_test1, t_test6).reshape(-1,1)
#                     #my_network = my_network.to(device)
#                     u_test_pred6 = my_network(test6).reshape(-1,1).to('cpu')
                    
#                     # Compute the relative L2 error norm (generalization error)
#                     error_6[epoch, :] = torch.mean((u_test_pred6 - u_test6)**2)/torch.mean(u_test6**2)
                    
                    # Set the model back to training mode
                    model.train()
                

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
# ## saving model
# print("new",wt_1)
start_time = time.time()
n_epochs = 10000
history = fit(my_network, training_set, interior, n_epochs, optimizer_, p=2, verbose=True )
end_time = time.time()
total_time = end_time - start_time
print("Training time: {:.2f} seconds".format(total_time))

# with open('causal_eb1.pkl', 'wb') as f:
#     pickle.dump(history, f)

# f.close()

# model_state_dict = my_network.state_dict()

# # Save the model state dictionary to a file
# torch.save(model_state_dict, 'causal_eb1.pth')

torch.save({
    'wt_1': wt_1,
    'wt_2': wt_2,
    'wt_3': wt_3,
    'wt_4': wt_4,
    'wt_5': wt_5,
    #'wt_6': wt_6,
}, 'weights_new.pth')

torch.save({
    'error_1': error_1,
    'error_2': error_2,
    'error_3': error_3,
    'error_4': error_4,
    'error_5': error_5,
   # 'error_6': error_6,
}, 'error_new.pth')




# # # Load the weights from the file
# loaded_weights = torch.load('weights.pth')

# loaded_wt_1 = loaded_weights['wt_1']
# loaded_wt_2 = loaded_weights['wt_2']
# loaded_wt_3 = loaded_weights['wt_3']
# loaded_wt_4 = loaded_weights['wt_4']
# loaded_wt_5 = loaded_weights['wt_5']
# loaded_wt_6 = loaded_weights['wt_6']

# loading model

x_test = torch.linspace(0, 8*pi, 10000).reshape(-1,1)
t_test = torch.ones((10000,1))
test = torch.cat([x_test, t_test],1)
u_test = exact_solution(x_test, t_test).reshape(-1,1)
my_network = my_network.cpu()
u_test_pred = my_network(test).reshape(-1,1)

# Compute the relative L2 error norm (generalization error)
relative_error_test = torch.mean((u_test_pred - u_test)**2)/torch.mean(u_test**2)
print("Relative Error Test: ", relative_error_test.detach().numpy()*100, "%")



