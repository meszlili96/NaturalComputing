import torch

"""Samples noise from unifrom distribution for Generator
"""
def sample_noise(size):
    noise = -1 * torch.rand(size, 2) + 0.5
    return noise