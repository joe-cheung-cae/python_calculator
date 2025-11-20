import numpy as np
from scipy import stats
from scipy.stats import weibull_min
import csv
import os
import sys
import argparse
import matplotlib.pyplot as plt

def to_weibull_pit(data, c, scale=1):
    """
    使用概率积分变换转换为Weibull分布

    Parameters:
    data: 原始数据 (numpy array)
    c: Weibull形状参数
    scale: Weibull尺度参数

    Returns:
    weibull_data: 转换后的Weibull数据
    """
    # 估计原始数据的经验CDF
    n = len(data)
    # Get ranks
    sorted_indices = np.argsort(data)
    ranks = np.empty(n, dtype=int)
    ranks[sorted_indices] = np.arange(1, n + 1)
    # Empirical CDF values: (rank - 0.5) / n
    uniform_data = (ranks - 0.5) / n

    # 使用Weibull分布的逆CDF
    weibull_data = weibull_min.ppf(uniform_data, c, scale=scale)
    return weibull_data

def read_raw_data_csv(csv_file):
    """
    Read raw data from CSV file.
    Expected format: single column of numeric values

    Parameters:
    csv_file (str): Path to input CSV file

    Returns:
    np.array: data values
    """
    data = []

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 1:
                    try:
                        value = float(row[0])
                        data.append(value)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid row {row}: {e}")
                        continue

        if len(data) == 0:
            raise ValueError("No valid data found in CSV file")

        return np.array(data)

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{csv_file}' not found")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

def write_data_csv(data, output_file):
    """
    Write data to CSV file.

    Parameters:
    data (np.array): data to write
    output_file (str): Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['value'])
            for value in data:
                writer.writerow([value])

        print(f"Transformed data exported to {output_file}")

    except Exception as e:
        raise Exception(f"Error writing to output file: {e}")

def plot_data_comparison(original_data, transformed_data, input_file, output_file, c, scale):
    """
    Plot histograms of original and transformed data for comparison.
    Includes Weibull PDF curve on transformed data.

    Parameters:
    original_data, transformed_data: numpy arrays
    input_file, output_file: file paths for plot title
    c, scale: Weibull parameters
    """
    plt.figure(figsize=(12, 6))

    # Plot original data histogram
    plt.subplot(1, 2, 1)
    plt.hist(original_data, bins=50, alpha=0.7, color='blue', density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Original Data\n{input_file}')
    plt.grid(True, alpha=0.3)

    # Plot transformed data histogram
    plt.subplot(1, 2, 2)
    plt.hist(transformed_data, bins=50, alpha=0.7, color='red', density=True, label='Transformed Data')

    # Add Weibull PDF curve
    x_pdf = np.linspace(0, np.max(transformed_data) * 1.2, 1000)
    weibull_pdf = weibull_min.pdf(x_pdf, c=c, scale=scale)
    plt.plot(x_pdf, weibull_pdf, 'k-', linewidth=2, label=f'Weibull PDF (c={c}, scale={scale})')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Transformed Weibull Data\nc={c}, scale={scale}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f'Data Transformation Comparison\nInput: {input_file} → Output: {output_file}', fontsize=14, y=1.02)

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
    parser = argparse.ArgumentParser(description='Transform raw data to Weibull distribution using Probability Integral Transform (PIT)')
    parser.add_argument('input_file', help='Path to input CSV file with raw data (single column)')
    parser.add_argument('output_file', help='Path to output CSV file for transformed Weibull data')
    parser.add_argument('--c', type=float, required=True, help='Weibull shape parameter (c)')
    parser.add_argument('--scale', type=float, default=1.0, help='Weibull scale parameter (default: 1.0)')
    parser.add_argument('--plot', action='store_true', help='Generate comparison histograms of input vs output data')

    args = parser.parse_args()

    try:
        print(f"Reading raw data from {args.input_file}...")

        # Read input data
        original_data = read_raw_data_csv(args.input_file)
        print(f"Loaded {len(original_data)} data points")

        # Apply PIT transformation
        print(f"Applying Weibull PIT transformation with c={args.c}, scale={args.scale}...")
        transformed_data = to_weibull_pit(original_data, args.c, args.scale)

        # Export transformed data
        write_data_csv(transformed_data, args.output_file)

        print("Transformation completed successfully!")

        # Generate comparison plot if requested
        if args.plot:
            plot_data_comparison(original_data, transformed_data, args.input_file, args.output_file, args.c, args.scale)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()