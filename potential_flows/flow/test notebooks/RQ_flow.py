import sys
sys.path.append("../../")
from potential_flows import transforms
from potential_flows import potential
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

n = 1000
d = 2
x1 = torch.randn(n,1)
x2 = x1.pow(2) + torch.randn(n,1)
from_data = torch.cat([x1, x2], dim=1)
to_data = torch.randn(n,d)


layer_identity = transforms.IdentityTransform()
layer_affine1 = transforms.PositiveLinear(d)
layer_spline1 = transforms.RQspline(data_shape=(d,), num_bins=4)
layer_affine2 = transforms.PositiveLinear(d)
layer_spline2 = transforms.RQspline(data_shape=(d,), num_bins=4)
layers = transforms.CompositeTransform([layer_affine1, layer_spline1, layer_affine2, layer_spline2])
potential_flow = potential.ICRQ(layers, (d,))

def objective_function(x, y, potential_flow):
    '''
    y: n * d
    x: n * d
    potential_flow: potential function
    '''
    return -torch.sum(torch.sum(x*y, dim=-1) - potential_flow.integral(x))

def approx_inv_flow(potential_flow, y):
    x0 = torch.zeros_like(y, requires_grad=True)
    optimizer = torch.optim.Adam([x0], lr=.001)
    def closure():
        optimizer.zero_grad()  # Clear previous gradients
        loss = objective_function(x0, y, potential_flow)
        loss.backward()  # Compute gradients
        return loss
    max_iter = 1000
    for i in tqdm(range(max_iter)):
        optimizer.step(closure)
    optimized_parameters = x0.detach()
    return optimized_parameters

def calculate_loss_gradients(potential_flow, x, y):
    n = x.shape[0]
    with torch.no_grad():
        x_star = approx_inv_flow(potential_flow, y)
    loss = torch.mean(potential_flow.integral(x) + torch.sum(y*x_star, dim=-1) - potential_flow.integral(x_star))
    print(f'Loss: {loss}')
    x.requires_grad = True
    x_star.requires_grad = True
    grad = torch.sum(potential_flow.integral(x))/n
    grad_conjugate = torch.sum(potential_flow.integral(x_star))/n

    model_gradient = torch.autograd.grad(grad, potential_flow.parameters())
    model_star_gradient = torch.autograd.grad(grad_conjugate, potential_flow.parameters())

    return loss, tuple(a - b for a, b in zip(model_gradient, model_star_gradient))


max_iter = 200
learning_rate = 1e-3
losses = []
for epoch in tqdm(range(max_iter)):
    potential_flow.zero_grad()
    loss, loss_gradient = calculate_loss_gradients(potential_flow, from_data, to_data)
    losses.append(loss)
    # Update model parameters
    with torch.no_grad():
        for param, grad in zip(potential_flow.parameters(), loss_gradient):
            param -= learning_rate * grad

model_path = "potential_flow.pth"
torch.save(potential_flow.state_dict(), model_path)
