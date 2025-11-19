# Weibull Distribution Visualizer and Population Balance Model

## Description

This project provides tools for visualizing Weibull probability distributions and simulating particle aggregation using Population Balance Models (PBM). The Weibull distribution is commonly used in reliability engineering, survival analysis, and particle size distribution modeling. The PBM implementation uses the Smoluchowski equation to model how particle size distributions evolve through aggregation processes.

## Features

### Weibull Visualizer (`weibull_visualizer.py`)
- Interactive visualization of Weibull distributions
- Adjustable shape (k) and scale (λ) parameters via sliders
- Real-time PDF plotting
- Export functionality to CSV format
- Customizable parameter ranges

### Weibull Overlay Visualizer (`weibull_overlay_visualizer.py`)
- Visualize and overlay two Weibull distributions
- Adjust peak coordinates for each distribution
- Weighted combination of distributions
- Automatic parameter calculation from peak coordinates
- Export combined data to CSV

### Population Balance Model (`pbm_model.py`)
- Implementation of Smoluchowski aggregation equation
- Time evolution of particle size distributions
- Support for different aggregation kernels
- Import initial distributions from CSV
- Export evolved distributions

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Usage

### Weibull Visualizer

Run the interactive Weibull distribution visualizer:

```bash
python weibull_visualizer.py
```

This opens an interactive window where you can:
- Adjust shape parameter (k) and scale parameter (λ) using sliders
- Modify slider ranges using text boxes
- Export the current distribution to `weibull_data.csv`

### Weibull Overlay Visualizer

Run the overlay visualizer for two distributions:

```bash
python weibull_overlay_visualizer.py
```

Features:
- Adjust peak coordinates for two Weibull distributions
- Control the weighting between distributions
- View individual distributions and their weighted sum
- Export data to `weibull_overlay_data.csv`

### Population Balance Model

The PBM can be run as a script or imported as a module:

```bash
python pbm_model.py
```

This will:
- Load initial distribution from `weibull_data.csv`
- Evolve the distribution over time using aggregation
- Export final distribution to `evolved_distribution.csv`
- Generate visualization in `pbm_evolution.png`

## Examples

### Creating Initial Distribution

1. Run `python weibull_visualizer.py`
2. Adjust parameters (e.g., k=2.0, λ=1.414)
3. Click "Export to CSV" to create `weibull_data.csv`

### Running PBM Simulation

```python
from pbm_model import PopulationBalanceModel
import numpy as np

# Load initial distribution
pbm = PopulationBalanceModel('weibull_data.csv')

# Evolve over time
t = np.linspace(0, 1, 11)
evolved_distributions = pbm.evolve(t)

# Export final distribution
pbm.export_distribution(evolved_distributions[-1], 'final_distribution.csv')
```

## File Descriptions

- `weibull_visualizer.py`: Interactive single Weibull distribution visualizer
- `weibull_overlay_visualizer.py`: Interactive two-distribution overlay visualizer
- `pbm_model.py`: Population Balance Model implementation
- `weibull_data.csv`: Example/initial Weibull distribution data
- `evolved_distribution.csv`: Output from PBM simulation
- `pbm_evolution.png`: Visualization of PBM evolution
- `.gitignore`: Git ignore file for Python projects

## Mathematical Background

### Weibull Distribution

The Weibull distribution PDF is given by:

f(x; k, λ) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)

Where:
- k: shape parameter
- λ: scale parameter

### Population Balance Model

The PBM uses the Smoluchowski equation for aggregation:

dn/dt = 1/2 ∫∫ K(v,u) n(v) n(u) dv du - ∫ K(v,u) n(v) n(u) du

Where n(v) is the number density of particles of volume v, and K is the aggregation kernel.

This implementation uses a discretized version with constant kernel for simplicity.