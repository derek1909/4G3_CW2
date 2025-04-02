import torch
import math
import matplotlib.pyplot as plt
import os
from rnn import RNNModel

# Specify results folder
result_folder = "results"
os.makedirs(result_folder, exist_ok=True)

# Simulation parameters
T = 0.06         # total simulation time in seconds (60 ms)
noise_std = 1.0  # noise standard deviation
theta = math.pi  # stimulus orientation
trials = 100     # number of repeated trials

# List of models to simulate (1, 2, 3, 4)
model_types = [1, 3, 4]

plt.figure()
for model_type in model_types:
    # Instantiate the model using default parameters
    model = RNNModel(model_type=model_type, device='cpu')
    m = model.m  # number of output neurons
    
    # Use the simulate function to run the dynamics
    t_record, r_record = model.simulate(theta, T=T)
    output_noisy = model.readout(r_record, trials=trials, noise_std=noise_std)
    decoded = model.decode(output_noisy)
    
    # Convert t_record (in seconds) to a time axis in milliseconds
    time_axis = t_record.cpu().numpy() * 1000
    
    error = model.circular_distance(decoded, theta)
    error_mean = error.mean(dim=1)
        
    # Plot the decoding error time course for the current model
    plt.plot(time_axis, error_mean, label=f"Model {model_type}")

plt.xlabel("Time (ms)")
plt.ylabel("Decoding Error (radians)")
plt.yscale("log")
plt.title("Decoding Error Time Course for All Models")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(result_folder, "all_models_decoding_error.png"), dpi=300)
plt.close()