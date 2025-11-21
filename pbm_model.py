import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

class GrindingMillScaleUp:
    """
    Grinding mill scale-up design class based on population balance model
    """
    
    def __init__(self):
        self.mill_params = {}
        self.breakage_params = {}
        self.classification_params = {}
        
    def set_mill_parameters(self, diameter, length, speed, filling_ratio,
                          feed_rate, solid_density, pulp_density):
        """Set basic mill parameters"""
        self.mill_params = {
            'diameter': diameter,
            'length': length,
            'speed': speed,
            'filling_ratio': filling_ratio,
            'feed_rate': feed_rate,
            'solid_density': solid_density,
            'pulp_density': pulp_density
        }
        
    def set_breakage_parameters(self, alpha, beta, gamma, phi=0.6):
        """
        Set breakage function parameters

        Parameters:
        alpha, beta, gamma: Selection function parameters
        phi: Breakage function parameter
        """
        self.breakage_params = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'phi': phi
        }
        
    def set_classification_parameters(self, d50, sharpness):
        """Set classification function parameters"""
        self.classification_params = {
            'd50': d50,
            'sharpness': sharpness
        }
        
    def selection_function(self, particle_size):
        """
        Calculate selection function (breakage rate function)

        Parameters:
        particle_size: Particle size (mm)

        Returns:
        Selection function value (1/min)
        """
        if not self.breakage_params:
            raise ValueError("Breakage parameters not set. Call set_breakage_parameters first.")

        alpha = self.breakage_params['alpha']
        beta = self.breakage_params['beta']
        gamma = self.breakage_params['gamma']

        # Selection function based on Austin model
        if particle_size <= 0:
            return 0.0

        S = (alpha * particle_size**gamma) / (1 + (particle_size / beta)**gamma)
        return S
    
    def improved_selection_function(self, particle_size, scale_factor=1.0):
        """
        Improved selection function considering scale effects

        Parameters:
        particle_size: Particle size (mm)
        scale_factor: Scale correction factor
        """
        alpha = self.breakage_params['alpha']
        beta = self.breakage_params['beta']
        gamma = self.breakage_params['gamma']

        # Base selection function
        S_base = (alpha * particle_size**gamma) / (1 + (particle_size / beta)**gamma)

        # If mill parameters not set, use base selection function
        if not self.mill_params:
            return S_base

        # Scale effect correction
        diameter = self.mill_params['diameter']
        scale_correction = diameter**0.2  # Empirical correction factor

        # Operating condition correction
        speed_ratio = self.mill_params['speed'] / 20.0  # Relative to reference speed
        filling_ratio = self.mill_params['filling_ratio']

        S_corrected = S_base * scale_correction * speed_ratio * filling_ratio / 0.3

        return S_corrected
    
    def breakage_function(self, x, y):
        """
        Calculate breakage function (breakage distribution function)

        Parameters:
        x: Particle size after breakage (mm)
        y: Particle size before breakage (mm)

        Returns:
        Breakage function value
        """
        if x >= y or y <= 0:
            return 0.0

        if not self.breakage_params:
            raise ValueError("Breakage parameters not set. Call set_breakage_parameters first.")

        phi = self.breakage_params['phi']
        gamma = self.breakage_params['gamma']

        # Breakage function based on Broadbent-Calcott model
        B_xy = phi * ((1 - np.exp(-(x/y)**gamma)) / (1 - np.exp(-1)))
        return B_xy
    
    def classification_function(self, particle_size):
        """Calculate classification function"""
        if not self.classification_params:
            return 1.0  # If classification parameters not set, pass all by default

        d50 = self.classification_params['d50']
        sharpness = self.classification_params['sharpness']

        if particle_size <= 0:
            return 1.0

        # Classification function based on Plitt model
        C = 1 / (1 + (particle_size / d50)**sharpness)
        return C
    
    def calculate_residence_time(self):
        """Calculate average residence time"""
        if not self.mill_params:
            raise ValueError("Mill parameters not set. Call set_mill_parameters first.")

        mill_diameter = self.mill_params['diameter']
        mill_length = self.mill_params['length']
        filling_ratio = self.mill_params['filling_ratio']
        feed_rate = self.mill_params['feed_rate']
        pulp_density = self.mill_params['pulp_density']

        # Calculate effective mill volume
        mill_volume = np.pi * (mill_diameter/2)**2 * mill_length * filling_ratio

        # Calculate pulp volume flow rate (m³/s)
        pulp_volume_flow = (feed_rate * 1000 / 3600) / pulp_density

        if pulp_volume_flow <= 0:
            return 60.0  # Default value

        # Calculate residence time (min)
        residence_time = (mill_volume / pulp_volume_flow) / 60
        return max(residence_time, 1.0)  # Minimum residence time 1 minute
    
    def population_balance_model(self, m, t, feed_size_dist, size_classes):
        """
        Population balance equation

        Parameters:
        m: Mass fractions of each size class
        t: Time
        feed_size_dist: Feed size distribution
        size_classes: Size classes

        Returns:
        dm/dt Population balance equation derivative
        """
        n_sizes = len(size_classes)
        dmdt = np.zeros(n_sizes)

        # Calculate residence time
        residence_time = self.calculate_residence_time()

        for i in range(n_sizes):
            # Breakage loss term - using improved selection function
            S_i = self.improved_selection_function(size_classes[i])
            loss_term = -S_i * m[i]

            # Breakage generation term
            generation_term = 0.0
            for j in range(i):
                S_j = self.improved_selection_function(size_classes[j])
                B_ij = self.breakage_function(size_classes[i], size_classes[j])
                generation_term += B_ij * S_j * m[j]

            # Feed term and discharge term
            feed_term = feed_size_dist[i] / residence_time
            discharge_term = -m[i] / residence_time

            dmdt[i] = loss_term + generation_term + feed_term + discharge_term

        return dmdt
    
    
    def simulate_grinding_circuit(self, feed_size_dist, size_classes,
                                simulation_time=60, time_steps=200):
        """Simulate grinding circuit"""
        # Initial conditions
        m0 = feed_size_dist.copy()

        # Time points
        t = np.linspace(0, simulation_time, time_steps)

        # Solve population balance equation
        solution = odeint(self.population_balance_model, m0, t,
                         args=(feed_size_dist, size_classes))

        return t, solution
    
    def simulate_with_improved_parameters(self, feed_size_dist, size_classes):
        """Simulate with improved parameters"""
        # Calculate scale-related correction factor
        diameter = self.mill_params['diameter']
        scale_factor = (diameter / 0.3)**0.15  # Gentle scale correction

        # Update breakage parameters considering scale effects
        original_alpha = self.breakage_params['alpha']
        self.breakage_params['alpha'] = original_alpha * scale_factor

        # Run simulation
        t, solution = self.simulate_grinding_circuit(feed_size_dist, size_classes)

        # Restore original parameters
        self.breakage_params['alpha'] = original_alpha

        return t, solution
    
    def optimize_mill_parameters(self, target_product_size, feed_size_dist,
                               size_classes, initial_guess):
        """Optimize mill operating parameters"""
        def objective_function(params):
            speed, filling_ratio, feed_rate = params

            # Update mill parameters
            self.mill_params['speed'] = max(speed, 1.0)
            self.mill_params['filling_ratio'] = max(min(filling_ratio, 0.5), 0.1)
            self.mill_params['feed_rate'] = max(feed_rate, 0.1)

            try:
                # Run simulation
                t, product_dist = self.simulate_grinding_circuit(
                    feed_size_dist, size_classes, simulation_time=30, time_steps=100)

                # Calculate objective function (mass fraction of target size class in product)
                final_product = product_dist[-1, :]
                target_fraction = np.sum(final_product[size_classes <= target_product_size])

                # Maximize target size class yield
                return -target_fraction
            except:
                return 1e6  # Return large value if simulation fails

        # Set constraints
        bounds = [
            (10, 30),      # Speed range (rpm)
            (0.2, 0.4),    # Filling ratio range
            (50, 200)      # Feed rate range (t/h)
        ]

        # Optimize
        result = minimize(objective_function, initial_guess,
                         method='SLSQP', bounds=bounds)

        return result
    
    def plot_size_distribution(self, t, solution, size_classes,
                               specific_times=None, title="Size Distribution Evolution", save_path=None):
        """Plot size distribution evolution over time"""
        if specific_times is None:
            specific_times = [t[0], t[len(t)//4], t[len(t)//2], t[3*len(t)//4], t[-1]]

        plt.figure(figsize=(10, 6))

        for time_point in specific_times:
            idx = np.argmin(np.abs(t - time_point))
            plt.semilogx(size_classes, solution[idx, :],
                        label=f'Time = {time_point:.1f} min', linewidth=2)

        plt.xlabel('Particle Size (mm)')
        plt.ylabel('Mass Fraction')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()






def comprehensive_comparison():
    """Comprehensive comparison of laboratory and industrial mill performance"""

    # Create improved mill design object
    improved_mill = GrindingMillScaleUp()

    # Set common breakage parameters
    improved_mill.set_breakage_parameters(
        alpha=1.2, beta=2.5, gamma=0.8, phi=0.6
    )

    improved_mill.set_classification_parameters(d50=0.1, sharpness=2.5)

    # Size distribution - normal distribution for realistic feed
    size_classes = np.array([100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005])

    # Generate normal distribution centered around 1mm
    mean_size = 1.0  # Mean particle size in mm
    std_size = 0.8   # Standard deviation

    # Calculate normal distribution values
    normal_values = np.exp(-0.5 * ((size_classes - mean_size) / std_size) ** 2)
    normal_values /= normal_values.sum()  # Normalize

    feed_size_dist = normal_values

    # Display the normal feed size distribution
    print("Normal feed size distribution (centered at 1mm):")
    for i, size in enumerate(size_classes):
        print(f"Size {size} mm: {feed_size_dist[i]:.4f}")
    print()

    # Validate parameters and functions
    print("Validating selection function:")
    for size in size_classes:
        S = improved_mill.improved_selection_function(size)
        print(f"Size {size} mm: S = {S:.4f} 1/min")

    print("\nValidating breakage function:")
    for i in range(len(size_classes)):
        for j in range(i):
            B = improved_mill.breakage_function(size_classes[i], size_classes[j])
            print(f"B({size_classes[i]:.3f}, {size_classes[j]:.3f}) = {B:.4f}")

    # Laboratory mill simulation
    print("\n=== Laboratory Mill Simulation ===")
    improved_mill.set_mill_parameters(
        diameter=0.3, length=0.3, speed=20,
        filling_ratio=0.3, feed_rate=0.1,
        solid_density=2700, pulp_density=1600
    )

    t_lab, product_lab = improved_mill.simulate_with_improved_parameters(
        feed_size_dist, size_classes)
    print("Laboratory simulation completed!")

    # Display final product size distribution
    final_product = product_lab[-1, :]
    print("\nLaboratory final product size distribution (60 min):")
    for i, size in enumerate(size_classes):
        print(f"Size {size} mm: {final_product[i]:.4f}")

    # Run simulations for different grinding times
    grinding_times = [20, 40, 60]
    print("\n=== Simulations for Different Grinding Times ===")
    for sim_time in grinding_times:
        t_temp, product_temp = improved_mill.simulate_grinding_circuit(
            feed_size_dist, size_classes, simulation_time=sim_time)
        final_temp = product_temp[-1, :]
        print(f"\nGrinding time: {sim_time} min")
        print("Final product size distribution:")
        for i, size in enumerate(size_classes):
            print(f"Size {size} mm: {final_temp[i]:.4f}")

    # Optimize mill parameters
    print("\n=== Mill Parameter Optimization ===")
    initial_params = [20, 0.3, 0.1]  # Initial guess
    target_size = 0.074  # Target particle size 74μm

    try:
        optimization_result = improved_mill.optimize_mill_parameters(
            target_size, feed_size_dist, size_classes, initial_params)

        print(f"Optimization results:")
        print(f"Optimal speed: {optimization_result.x[0]:.1f} rpm")
        print(f"Optimal filling ratio: {optimization_result.x[1]:.3f}")
        print(f"Optimal feed rate: {optimization_result.x[2]:.1f} t/h")
        print(f"Objective function value: {-optimization_result.fun:.3f}")

    except Exception as e:
        print(f"Optimization failed: {e}")

    return improved_mill, t_lab, product_lab

def run_complete_demonstration():
    """Run laboratory mill performance demonstration"""
    print("=" * 60)
    print("Laboratory Mill Performance Demonstration")
    print("=" * 60)

    # Run laboratory mill simulation
    print("\nLaboratory Mill Performance Analysis")
    improved_mill, t_lab, product_lab = comprehensive_comparison()

    # Plot size distribution at specific times
    print("\nSize Distribution Plot at 0, 1, 2, 4, 10, 20 minutes")
    improved_mill.plot_size_distribution(t_lab, product_lab,
                                         np.array([100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]),
                                         specific_times=[0, 1, 2, 4, 10, 20],
                                         title="Laboratory Mill Size Distribution at 0, 1, 2, 4, 10, 20 min",
                                         save_path="output/laboratory_mill_size_distribution.png")

    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)

# 运行完整演示
if __name__ == "__main__":
    run_complete_demonstration()