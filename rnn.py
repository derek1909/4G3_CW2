import torch
import math

class RNNModel(torch.nn.Module):
    r"""
    A recurrent neural network (RNN) model that simulates V1 dynamics using one of four connectivity models.
    
    The dynamics are defined by:
    \begin{align}
    \tau \frac{dr}{dt} &= -r + W\, r + B\, h(\theta)\,\delta(t)
    \end{align}
    where \(r(t) \in \mathbb{R}^n\) is the firing rate vector, \(W\) is the recurrent connectivity matrix,
    \(B\) is the feedforward weight matrix, and \(h(\theta) \in \mathbb{R}^m\) is given by:
    \begin{align}
    h_i(\theta) &= V(\phi_i - \theta), \quad V(z) = \exp\left(\frac{\cos(z)-1}{\kappa^2}\right)
    \end{align}
    with \(\phi_i = \frac{2\pi i}{m}\) for \(i=0,\ldots,m-1\).
    
    The models are:
      - Model 1: No recurrence, \(W = 0\).
      - Model 2: Random symmetric connectivity, 
        \(\displaystyle W^{(2)} = R\Big(\tilde{W} + \tilde{W}^T,\alpha\Big)\) with \(\tilde{W}_{ij} \sim \mathcal{N}(0,1)\).
      - Model 3: Symmetric ring structure, 
        \(\displaystyle W^{(3)} = R\Big(W_{\text{ring}},\alpha\Big)\) with \(W_{\text{ring},ij} = V(\phi_i - \phi_j)\).
      - Model 4: Balanced ring structure with \(n=2m\), 
        \(\displaystyle W^{(4)} = \begin{pmatrix} \tilde{W} & -\tilde{W} \\ \tilde{W} & -\tilde{W} \end{pmatrix}\),
        where \(\tilde{W} = R\Big(W_{\text{ring}},\alpha'\Big)\). For Model 4 the input and output
        weights are chosen as:
        \begin{align}
        B &= \begin{pmatrix} I_m \\ 0_m \end{pmatrix}, \quad
        C = \begin{pmatrix} I_m & 0_m \end{pmatrix}.
        \end{align}
    
    Parameters:
      - model_type (int): 1, 2, 3, or 4 specifying the connectivity model.
      - m (int): Number of orientations (and neurons for models 1,2,3; for Model 4, \(n=2m\)).
      - tau (float): Neuronal time constant (in seconds, default 0.02 s i.e. 20 ms).
      - dt (float): Time step for Euler integration (default 0.001 s).
      - alpha (float): Scaling parameter for models 2 and 3 (default 0.9).
      - alpha_prime (float): Scaling parameter for Model 4 (default 0.9).
      - kappa (float): Parameter in the activation function \(V(z)\) (default \(\pi/4\)).
      - device (str): Torch device ('cpu' or 'cuda').
    """
    def __init__(self, model_type=1, m=200, tau=0.02, dt=0.001, alpha=0.9, alpha_prime=0.9, 
                 kappa=math.pi/4, device='cpu'):
        super(RNNModel, self).__init__()
        self.model_type = model_type
        self.m = m
        # For models 1,2,3: n = m; for model 4: n = 2m.
        if model_type in [1, 2, 3]:
            self.n = m
        elif model_type == 4:
            self.n = 2 * m
        else:
            raise ValueError("model_type must be 1, 2, 3, or 4")
            
        self.tau = tau
        self.dt = dt
        self.alpha = alpha
        self.alpha_prime = alpha_prime
        self.kappa = kappa
        self.device = device
        
        # Create a grid of orientations: φ_i = 2π i/m for i = 0,..., m-1.
        self.phi = 2 * math.pi * torch.arange(m, dtype=torch.float32, device=self.device) / m
        
        # Build connectivity matrix W and the matrices B and C.
        self.W = self.build_W()
        self.B, self.C = self.build_BC()

    def V(self, z):
        return torch.exp((torch.cos(z) - 1) / (self.kappa ** 2))
    
    def scale_W(self, W, target_alpha):
        eigenvalues = torch.linalg.eigvals(W)
        max_real = eigenvalues.real.max()
        return W * (target_alpha / max_real)
    
    def build_W(self):
        if self.model_type == 1:
            # Model 1: No recurrence.
            return torch.zeros((self.n, self.n), dtype=torch.float32, device=self.device)
        
        elif self.model_type == 2:
            # Model 2: Random symmetric connectivity.
            tildeW = torch.randn((self.n, self.n), dtype=torch.float32, device=self.device)
            W_unscl = tildeW + tildeW.t()
            W = self.scale_W(W_unscl, self.alpha)
            return W
        
        elif self.model_type == 3:
            # Model 3: Symmetric ring structure.
            diff = self.phi.unsqueeze(1) - self.phi.unsqueeze(0)  # shape (m, m) #CHECK THIS
            W_ring = self.V(diff)
            W = self.scale_W(W_ring, self.alpha)
            return W
        
        elif self.model_type == 4:
            # Model 4: Balanced ring structure.
            # First compute the ring connectivity as in Model 3.
            diff = self.phi.unsqueeze(1) - self.phi.unsqueeze(0)
            W_ring = self.V(diff)
            W_tilde = self.scale_W(W_ring, self.alpha_prime)
            # Construct block matrix: [W_tilde, -W_tilde; W_tilde, -W_tilde].
            top = torch.cat([W_tilde, -W_tilde], dim=1)
            bottom = torch.cat([W_tilde, -W_tilde], dim=1)
            W = torch.cat([top, bottom], dim=0)
            return W
        
        else:
            raise ValueError("Invalid model type")
    
    def build_BC(self):
        if self.model_type in [1, 2, 3]:
            B = torch.eye(self.n, self.m, dtype=torch.float32, device=self.device)
            C = torch.eye(self.m, self.n, dtype=torch.float32, device=self.device)
            return B, C
        
        elif self.model_type == 4:
            I_m = torch.eye(self.m, dtype=torch.float32, device=self.device)
            zeros = torch.zeros((self.m, self.m), dtype=torch.float32, device=self.device)
            B = torch.cat([I_m, zeros], dim=0)  # Shape: (2m, m)
            C = torch.cat([I_m, zeros], dim=1)    # Shape: (m, 2m)
            return B, C
        
        else:
            raise ValueError("Invalid model type")
    
    def h(self, theta):
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=torch.float32, device=self.device)
        diff = self.phi - theta
        return self.V(diff)
    
    def simulate(self, theta, T=0.06):
        num_steps = int(T / self.dt)
        h_theta = self.h(theta)
        r = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        # Initialize tensors to record time and r values
        r_record = torch.zeros((num_steps + 1, self.n), device=self.device)
        t_record = torch.zeros(num_steps + 1, device=self.device)
        
        # Simulate the dynamics using Euler integration
        for step in range(0, num_steps + 1):
            t = step * self.dt
            dr = ( -r + self.W @ r + self.B @ h_theta * (t==0)/self.dt ) * (self.dt / self.tau)
            r = r + dr
            r_record[step] = r.clone().detach()
            t_record[step] = t

        return t_record, r_record
        
    def readout(self, r_record, trials=1, noise_std=1.0):
        # Generate noisy output for each trial at final time
        output_clean = torch.einsum('ij,tj->ti', self.C, r_record)  # Shape: (steps, m)
        output_noisy = output_clean.unsqueeze(-1) + torch.randn((output_clean.shape[0], self.m, trials), device=self.device) * noise_std
        return output_noisy

    def decode(self, output):
        """
        Decode the orientation (in radians) from the noisy output.
        output: (steps, m) or (steps, m, trials).
        theta_hat: 
        """
        # If output has shape (steps, m), add a trial dimension.
        if output.dim() == 2:
            output = output.unsqueeze(-1)  # Now shape is (steps, m, 1)
        
        # Reshape phi to be broadcastable: shape (1, m, 1)
        phi = self.phi.view(1, self.m, 1)
        
        # Compute numerator and denominator by summing over the m dimension
        num = (output * torch.sin(phi)).sum(dim=1)  # shape: (steps, trials)
        den = (output * torch.cos(phi)).sum(dim=1)    # shape: (steps, trials)
        theta_hat = torch.atan2(num, den)  # shape: (steps, trials)
        
        # If there is only one trial, remove the trial dimension
        if theta_hat.shape[-1] == 1:
            theta_hat = theta_hat.squeeze(-1)  # shape: (steps,)
        
        return theta_hat

    def circular_distance(self, theta1, theta2):
        return torch.acos(torch.cos(theta1 - theta2))