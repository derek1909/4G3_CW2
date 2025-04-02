import torch
import math
import matplotlib.pyplot as plt
import os
from rnn import RNNModel

result_folder = "results"
os.makedirs(result_folder, exist_ok=True)

# Define stimulus orientation and simulation time.
theta = math.pi  # stimulus orientation (radians)
T = 0.06         # total simulation time (60 ms)

# List of models (1,2,3,4)
model_types = [1, 2, 3, 4]

# Loop over each model, simulate, and plot responses.
for model_type in model_types:
    # Instantiate the model with default parameters.
    model = RNNModel(model_type=model_type, device='cpu')
    
    # Simulate the network dynamics.
    t_record, r_record = model.simulate(theta, T=T)
    
    # Plot the responses.
    plt.figure()
    plt.plot(r_record[0, :].cpu().numpy(), label=r"$t=0^+$")
    plt.plot(r_record[20, :].cpu().numpy(), label="20 ms")
    plt.plot(r_record[60, :].cpu().numpy(), label="60 ms")
    plt.title(f"Firing Rates for Model {model_type}")
    plt.xlabel("Neuron index")
    plt.ylabel("Firing rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, f"model_{model_type}_responses.png"), dpi=300)
    plt.close()