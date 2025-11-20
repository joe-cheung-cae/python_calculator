import scipy.stats
import numpy as np
import csv
import os
import sys
import argparse

def read_distribution_csv(csv_file):
    """
    Read distribution data from CSV file.
    Expected format: x,pdf columns

    Parameters:
    csv_file (str): Path to input CSV file

    Returns:
    tuple: (x_values, pdf_values) as numpy arrays
    """
    x_values = []
    pdf_values = []

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for row in reader:
                if len(row) >= 2:
                    try:
                        x = float(row[0])
                        pdf = float(row[1])
                        x_values.append(x)
                        pdf_values.append(pdf)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid row {row}: {e}")
                        continue

        if len(x_values) == 0:
            raise ValueError("No valid data found in CSV file")

        return np.array(x_values), np.array(pdf_values)

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{csv_file}' not found")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

def validate_distribution_data(x_values, pdf_values):
    """
    Validate the distribution data for fitting.

    Parameters:
    x_values (np.array): x values
    pdf_values (np.array): pdf values

    Returns:
    tuple: (x_values, pdf_values, shift_applied) - validated and sorted, with shift info
    """
    if len(x_values) != len(pdf_values):
        raise ValueError("x and pdf arrays must have the same length")

    if len(x_values) < 3:
        raise ValueError("Need at least 3 data points for reliable fitting")

    # Sort by x values
    sort_indices = np.argsort(x_values)
    x_sorted = x_values[sort_indices]
    pdf_sorted = pdf_values[sort_indices]

    # Handle negative x values by shifting
    shift_applied = 0.0
    if np.any(x_sorted < 0):
        shift_applied = -x_sorted.min() + 1e-6  # Shift to make minimum x = 1e-6
        x_sorted = x_sorted + shift_applied
        print(f"Warning: Input data contains negative x values. Applied shift of +{shift_applied:.6f} to make all x >= 0 for Weibull fitting.")

    if np.any(pdf_sorted < 0):
        raise ValueError("PDF values must be non-negative")

    # Check if PDF integrates to approximately 1 (rough check)
    dx = np.diff(x_sorted)
    if len(dx) > 0:
        integral_approx = np.sum(pdf_sorted[:-1] * dx)
        if integral_approx < 0.1:  # Very rough check
            print(f"Warning: PDF integral is very small ({integral_approx:.4f}). Data may not represent a valid probability distribution.")

    return x_sorted, pdf_sorted, shift_applied

def fit_weibull_distribution(x_values, pdf_values):
    """
    Fit Weibull distribution parameters using maximum likelihood estimation.

    Parameters:
    x_values (np.array): x values from input distribution
    pdf_values (np.array): pdf values from input distribution

    Returns:
    tuple: (shape_param, scale_param) - fitted Weibull parameters
    """
    # For MLE fitting, we need actual samples from the distribution
    # Since we have PDF values, we can generate samples using inverse transform sampling
    # or use the fact that we have the PDF and can use it for fitting

    # Method 1: Generate samples from the empirical distribution
    # We'll use the PDF values as weights to sample from x_values
    n_samples = max(1000, len(x_values) * 10)  # Generate enough samples

    # Normalize pdf_values to create a proper probability mass function
    pdf_norm = pdf_values / np.sum(pdf_values)

    # Sample from the empirical distribution
    sampled_indices = np.random.choice(len(x_values), size=n_samples, p=pdf_norm)
    samples = x_values[sampled_indices]

    # Filter out zero or negative samples (Weibull requires x > 0)
    samples = samples[samples > 0]

    if len(samples) < 10:
        raise ValueError("Insufficient positive samples for Weibull fitting")

    # Fit Weibull distribution using MLE
    try:
        # scipy.stats.weibull_min.fit returns (c, loc, scale)
        # where c is shape, loc is location (usually 0 for Weibull), scale is scale parameter
        shape_param, loc_param, scale_param = scipy.stats.weibull_min.fit(samples, floc=0)

        if shape_param <= 0 or scale_param <= 0:
            raise ValueError("Fitted parameters are invalid")

        print(".3f")
        return shape_param, scale_param

    except Exception as e:
        raise Exception(f"Failed to fit Weibull distribution: {e}")

def generate_weibull_pdf(x_values, shape_param, scale_param):
    """
    Generate Weibull PDF values over the given x range.

    Parameters:
    x_values (np.array): x values to evaluate PDF at
    shape_param (float): Weibull shape parameter (k)
    scale_param (float): Weibull scale parameter (λ)

    Returns:
    np.array: Weibull PDF values
    """
    return scipy.stats.weibull_min.pdf(x_values, c=shape_param, scale=scale_param)

def write_distribution_csv(x_values, pdf_values, output_file):
    """
    Write distribution data to CSV file.

    Parameters:
    x_values (np.array): x values
    pdf_values (np.array): pdf values
    output_file (str): Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'pdf'])
            for x, pdf in zip(x_values, pdf_values):
                writer.writerow([x, pdf])

        print(f"Weibull distribution data exported to {output_file}")

    except Exception as e:
        raise Exception(f"Error writing to output file: {e}")

def convert_distribution(input_file, output_file):
    """
    Convert input distribution to Weibull distribution.

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    print(f"Reading distribution data from {input_file}...")

    # Read input data
    x_original, pdf_values = read_distribution_csv(input_file)

    # Validate data and handle shifting
    x_shifted, pdf_values, shift_applied = validate_distribution_data(x_original, pdf_values)

    print(f"Loaded {len(x_shifted)} data points")
    print(".3f")

    # Fit Weibull parameters using shifted data
    print("Fitting Weibull distribution parameters...")
    shape_param, scale_param = fit_weibull_distribution(x_shifted, pdf_values)

    # Generate Weibull PDF over the original x range
    print("Generating Weibull distribution data...")
    # For output, use original x range but only compute Weibull PDF for x >= 0
    x_output = x_original.copy()
    weibull_pdf = np.zeros_like(x_output)

    # Only compute Weibull PDF for non-negative x values
    positive_mask = x_original >= 0
    if np.any(positive_mask):
        # Shift the positive x values for Weibull computation
        x_positive_shifted = x_original[positive_mask] + shift_applied
        weibull_pdf[positive_mask] = generate_weibull_pdf(x_positive_shifted, shape_param, scale_param)

    # Write output
    write_distribution_csv(x_output, weibull_pdf, output_file)

    print("Conversion completed successfully!")
    print(".3f")

def plot_comparison(x_input, pdf_input, x_output, pdf_output, input_file, output_file):
    """
    Plot input and output distributions for comparison with smoothed curves.

    Parameters:
    x_input, pdf_input: Input distribution data
    x_output, pdf_output: Output Weibull distribution data
    input_file, output_file: File paths for plot title
    """
    import matplotlib.pyplot as plt
    from scipy import interpolate

    plt.figure(figsize=(12, 8))

    # Create smoothed curves using interpolation
    if len(x_input) > 3:
        # Sort input data for interpolation
        sort_idx = np.argsort(x_input)
        x_input_sorted = x_input[sort_idx]
        pdf_input_sorted = pdf_input[sort_idx]

        # Create smooth interpolation for input
        try:
            f_input = interpolate.interp1d(x_input_sorted, pdf_input_sorted, kind='cubic',
                                         bounds_error=False, fill_value=0)
            x_smooth = np.linspace(x_input_sorted.min(), x_input_sorted.max(), 1000)
            pdf_input_smooth = f_input(x_smooth)

            # Plot smoothed input distribution
            plt.plot(x_smooth, pdf_input_smooth, 'b-', linewidth=3, label='Input Distribution (smoothed)', alpha=0.8)
        except:
            # Fallback to original plotting if interpolation fails
            plt.plot(x_input, pdf_input, 'b-', linewidth=2, label='Input Distribution', alpha=0.7)
    else:
        # Not enough points for smoothing
        plt.plot(x_input, pdf_input, 'b-', linewidth=2, label='Input Distribution', alpha=0.7)

    # For Weibull output, use the data as-is (it's already smooth)
    if len(x_output) > 3:
        # Sort output data
        sort_idx = np.argsort(x_output)
        x_output_sorted = x_output[sort_idx]
        pdf_output_sorted = pdf_output[sort_idx]

        # Plot Weibull distribution
        plt.plot(x_output_sorted, pdf_output_sorted, 'r--', linewidth=3, label='Fitted Weibull Distribution', alpha=0.9)
    else:
        plt.plot(x_output, pdf_output, 'r--', linewidth=2, label='Fitted Weibull Distribution', alpha=0.8)

    # Add original data points as markers for reference
    plt.scatter(x_input, pdf_input, c='blue', s=30, alpha=0.6, label='Input Data Points')
    plt.scatter(x_output, pdf_output, c='red', s=30, alpha=0.6, label='Weibull Data Points')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('PDF', fontsize=12)
    plt.title(f'Distribution Conversion Comparison\nInput: {input_file} → Output: {output_file}', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plot_file = output_file.replace('.csv', '_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_file}")

    # Show the plot (only if not running in headless mode)
    try:
        plt.show()
    except:
        print("Plot display not available in headless environment")

def main():
    parser = argparse.ArgumentParser(description='Convert any distribution to Weibull distribution')
    parser.add_argument('input_file', help='Path to input CSV file with x,pdf columns')
    parser.add_argument('output_file', help='Path to output CSV file for Weibull distribution')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plot of input vs output distributions')

    args = parser.parse_args()

    try:
        if args.plot:
            # Read input data for plotting
            x_input, pdf_input = read_distribution_csv(args.input_file)

            # Perform conversion
            convert_distribution(args.input_file, args.output_file)

            # Read output data for plotting
            x_output, pdf_output = read_distribution_csv(args.output_file)

            # Generate comparison plot
            plot_comparison(x_input, pdf_input, x_output, pdf_output, args.input_file, args.output_file)
        else:
            convert_distribution(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()