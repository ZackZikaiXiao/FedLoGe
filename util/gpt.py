import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Define your policy network architecture, for instance using fully connected layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class FusionModel:
    def __init__(self, l_head, g_head, policy_net):
        self.l_head = l_head
        self.g_head = g_head
        self.policy_net = policy_net

    def forward(self, x):
        l_out = self.l_head(x)
        g_out = self.g_head(x)
        state = torch.cat([l_out, g_out], dim=-1)
        action = self.policy_net(state)  # Generate action, i.e., the method of adjusting weights
        fused_output = action[0] * l_out + action[1] * g_out  # Adjust weights according to action
        return fused_output

    def update_weights(self, reward, optimizer):
        loss = -reward  # Define the loss function, here assuming that reward is a scalar
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



import torch.optim as optim

# Assuming l_head and g_head are your local and global classification heads
# Define your policy network
policy_net = PolicyNetwork(input_dim=l_head.out_features + g_head.out_features, action_dim=2)

fusion_model = FusionModel(l_head, g_head, policy_net)

# Define the optimizer
optimizer = optim.Adam(fusion_model.parameters())

# Define the reward. In actual application, the reward should be based on the performance of the fused model.
reward = torch.tensor(1.0)

# Training loop
for i in range(1000):  # 1000 is just an example, should be the number of your epochs
    # Assuming x is your input data
    x = torch.rand(1, input_dim)  # input_dim is the dimension of your input data

    optimizer.zero_grad()

    # Forward pass
    output = fusion_model(x)

    # Update weights
    fusion_model.update_weights(reward, optimizer)

# Save the trained model
torch.save(fusion_model.state_dict(), "fusion_model.pth")