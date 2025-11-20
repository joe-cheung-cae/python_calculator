import numpy as np
import csv
import os
import math

def create_normal_distribution_csv(filename, mu=0.0, sigma=1.0, n_points=1000):
    """Create a CSV file with normal distribution data"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = np.linspace(mu - 4*sigma, mu + 4*sigma, n_points)
    # Normal PDF formula: (1/(σ√(2π))) * exp(-((x-μ)²)/(2σ²))
    prefactor = 1.0 / (sigma * math.sqrt(2 * math.pi))
    pdf = prefactor * np.exp(-((x - mu)**2) / (2 * sigma**2))

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf'])
        for xi, pdfi in zip(x, pdf):
            writer.writerow([xi, pdfi])

    print(f"Created {filename} with normal distribution (μ={mu}, σ={sigma})")

def create_exponential_distribution_csv(filename, rate=1.0, n_points=1000):
    """Create a CSV file with exponential distribution data"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = np.linspace(0.001, 5.0/rate, n_points)  # Start from small positive value
    pdf = rate * np.exp(-rate * x)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf'])
        for xi, pdfi in zip(x, pdf):
            writer.writerow([xi, pdfi])

    print(f"Created {filename} with exponential distribution (rate={rate})")

def create_gamma_distribution_csv(filename, shape=2.0, scale=1.0, n_points=1000):
    """Create a CSV file with gamma distribution data"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = np.linspace(0.001, (shape + 3*math.sqrt(shape))*scale, n_points)
    # Gamma PDF: (1/Γ(k)) * (1/θ^k) * x^(k-1) * exp(-x/θ)
    # Using math.gamma for the gamma function
    pdf = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi > 0:
            pdf[i] = (1.0 / math.gamma(shape)) * (1.0 / (scale ** shape)) * (xi ** (shape - 1)) * math.exp(-xi / scale)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf'])
        for xi, pdfi in zip(x, pdf):
            writer.writerow([xi, pdfi])

    print(f"Created {filename} with gamma distribution (shape={shape}, scale={scale})")

if __name__ == "__main__":
    # Create test data
    create_normal_distribution_csv("input/normal_test_data.csv", mu=2.0, sigma=1.5)
    create_exponential_distribution_csv("input/exponential_test_data.csv", rate=0.5)
    create_gamma_distribution_csv("input/gamma_test_data.csv", shape=2.0, scale=1.0)