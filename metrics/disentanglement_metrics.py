import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LinearProbe(nn.Module):
    """A simple linear classifier for probing latent variables."""

    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_and_evaluate_probe(latents, labels, device):
    """
    Trains a linear probe on frozen latents and evaluates its accuracy.
    Args:
        latents (torch.Tensor): The frozen latent variables (N, D).
        labels (torch.Tensor): The ground truth labels (N,).
    Returns:
        float: The final accuracy of the trained probe.
    """
    num_classes = len(torch.unique(labels))
    probe = LinearProbe(latents.shape[1], num_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(latents, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print("  Training linear probe...")
    probe.train()
    for epoch in range(20):  # Train for a fixed number of epochs
        for z, y in loader:
            optimizer.zero_grad()
            y_hat = probe(z)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for z, y in loader:
            y_hat = probe(z)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f"  Linear probe accuracy: {accuracy:.2f}%")
    return accuracy