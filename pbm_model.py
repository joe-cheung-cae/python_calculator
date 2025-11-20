import numpy as np
import csv
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

class PopulationBalanceModel:
    def __init__(self, csv_file, kernel_type='constant', kernel_value=1.0):
        """
        Initialize PBM with initial size distribution from CSV.
        CSV should have columns: x, pdf
        """
        self.x = []
        self.pdf = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self.x.append(float(row[0]))
                self.pdf.append(float(row[1]))
        self.x = np.array(self.x)
        self.pdf = np.array(self.pdf)
        self.n_bins = len(self.x)
        self.dx = self.x[1] - self.x[0]  # assume uniform spacing

        # Treat pdf as initial number density n(x)
        self.n0 = self.pdf.copy()

        # Aggregation Kernel
        self.kernel_type = kernel_type
        self.kernel_value = kernel_value

    def kernel(self, i, j):
        """Aggregation kernel"""
        if self.kernel_type == 'constant':
            return self.kernel_value
        # Add other kernels if needed
        return self.kernel_value

    def dndt(self, n, t):
        """RHS of Smoluchowski equation for discretized system (aggregation only)"""
        if self.kernel_type == 'constant':
            K = self.kernel_value
            # Birth term: 0.5 * K * convolution of n with itself
            conv = np.convolve(n, n)[:self.n_bins]
            birth = 0.5 * K * conv
            # Death term: K * n[k] * sum(n)
            sum_n = np.sum(n)
            death = K * n * sum_n
            dn = birth - death
        else:
            # Fallback to loop for non-constant kernels
            dn = np.zeros_like(n)
            for k in range(self.n_bins):
                birth = 0.0
                for i in range(k+1):
                    j = k - i
                    if j < self.n_bins:
                        birth += self.kernel(i, j) * n[i] * n[j]
                birth *= 0.5
                death = 0.0
                for j in range(self.n_bins):
                    death += self.kernel(k, j) * n[k] * n[j]
                dn[k] = birth - death
        return dn

    def evolve(self, t_span):
        """
        Evolve the distribution over time t_span (array of times)
        """
        sol = odeint(self.dndt, self.n0, t_span)
        return sol

    def export_distribution(self, n, filename):
        """
        Export the distribution to CSV
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'n'])
            for i in range(self.n_bins):
                writer.writerow([self.x[i], n[i]])

# Example usage
if __name__ == "__main__":
    pbm = PopulationBalanceModel('weibull_data.csv')
    t = np.linspace(0, 1, 11)  # evolve from 0 to 1 time units
    sol = pbm.evolve(t)
    # Export final distribution
    pbm.export_distribution(sol[-1], 'output/evolved_distribution.csv')

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Initial distribution
    ax1.plot(pbm.x, pbm.n0, label='Initial Distribution', linewidth=2, color='blue')
    ax1.set_xlabel('Particle Size (x)')
    ax1.set_ylabel('Initial Number Density (n0)')
    ax1.set_title('Initial Particle Size Distribution')
    ax1.legend()
    ax1.grid(True)

    # Evolved distribution
    ax2.plot(pbm.x, sol[-1], label='Evolved Distribution (t=1)', linewidth=2, color='red')
    ax2.set_xlabel('Particle Size (x)')
    ax2.set_ylabel('Evolved Number Density (n)')
    ax2.set_title('Evolved Particle Size Distribution after Aggregation')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/pbm_evolution.png')
    plt.show()