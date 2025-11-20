import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import csv
import os
def weibull_peak_constant(k):
    """Compute the constant part of Weibull peak value for given shape parameter k"""
    if k <= 1:
        return np.inf  # Peak not defined for k <= 1
    ratio = (k - 1) / k
    return ratio ** ((k - 1) / k) * np.exp(-ratio ** (1 / k))

def compute_peak(k, lam):
    """Compute the peak coordinates (x, y) for given k and λ"""
    if k <= 1 or lam <= 0:
        return 0, 0
    ratio = (k - 1) / k
    x_peak = lam * ratio ** (1 / k)
    y_peak = (k / lam) * ratio ** ((k - 1) / k) * np.exp(-ratio ** (1 / k))
    return x_peak, y_peak

def find_params_for_peak(target_x, target_y, initial_guess=[2.0, 1.0]):
    """Find k and λ that achieve the target peak coordinates using optimization"""
    def objective(params):
        k, lam = params
        x, y = compute_peak(k, lam)
        return (x - target_x)**2 + (y - target_y)**2

    bounds = [(1.01, 10.0), (0.01, 10.0)]  # k >1, λ >0
    result = scipy.optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    if result.success:
        return result.x
    else:
        return initial_guess  # fallback


def update_plot(peak_x1, peak_y1, peak_x2, peak_y2, weight_k1=0.5, update_boxes=True):
    """Update the plot with new parameters for both distributions and their weighted sum"""
    if peak_x1 <= 0 or peak_y1 <= 0 or peak_x2 <= 0 or peak_y2 <= 0:
        return  # Skip invalid parameters

    # Find k and λ for distribution 1
    k1, lam1 = find_params_for_peak(peak_x1, peak_y1)

    # Find k and λ for distribution 2
    k2, lam2 = find_params_for_peak(peak_x2, peak_y2)

    # Clear both subplots
    ax1.clear()
    ax2.clear()

    # Use fixed step size of 0.1 for consistent data export
    x_max = 8.0  # Fixed maximum range
    x = np.arange(0, x_max + 0.1, 0.1)

    # Plot first distribution (blue)
    pdf1 = scipy.stats.weibull_min.pdf(x, c=k1, scale=lam1)
    ax1.plot(x, pdf1, 'b-', linewidth=2, label=f'Distribution 1 (k={k1:.3f}, λ={lam1:.3f})')

    # Plot second distribution (red)
    pdf2 = scipy.stats.weibull_min.pdf(x, c=k2, scale=lam2)
    ax1.plot(x, pdf2, 'r-', linewidth=2, label=f'Distribution 2 (k={k2:.3f}, λ={lam2:.3f})')

    # Calculate weighted sum for second subplot
    weight_k2 = 1.0 - weight_k1
    pdf_weighted_sum = weight_k1 * pdf1 + weight_k2 * pdf2

    # Plot the weighted sum in second subplot
    ax2.plot(x, pdf_weighted_sum, 'purple', linewidth=3, label=f'Weighted Sum (k1={weight_k1:.2f}, k2={weight_k2:.2f})')
    ax2.fill_between(x, pdf_weighted_sum, alpha=0.3, color='purple')

    # Calculate and plot peak lines for distribution 1
    if k1 > 1:
        peak_x1 = lam1 * ((k1 - 1) / k1) ** (1 / k1)
        peak_y1 = scipy.stats.weibull_min.pdf(peak_x1, c=k1, scale=lam1)
        ax1.axvline(x=peak_x1, color='b', linestyle='--', alpha=0.7, linewidth=1)
        ax1.axhline(y=peak_y1, color='b', linestyle='--', alpha=0.7, linewidth=1)
        ax1.plot([peak_x1], [peak_y1], 'bo', markersize=6, alpha=0.8)
        ax1.text(peak_x1 + 0.05, peak_y1 + 0.01, f'({peak_x1:.2f}, {peak_y1:.2f})', fontsize=9, color='b', alpha=0.8)

    # Calculate and plot peak lines for distribution 2
    if k2 > 1:
        peak_x2 = lam2 * ((k2 - 1) / k2) ** (1 / k2)
        peak_y2 = scipy.stats.weibull_min.pdf(peak_x2, c=k2, scale=lam2)
        ax1.axvline(x=peak_x2, color='r', linestyle='--', alpha=0.7, linewidth=1)
        ax1.axhline(y=peak_y2, color='r', linestyle='--', alpha=0.7, linewidth=1)
        ax1.plot([peak_x2], [peak_y2], 'ro', markersize=6, alpha=0.8)
        ax1.text(peak_x2 + 0.05, peak_y2 + 0.01, f'({peak_x2:.2f}, {peak_y2:.2f})', fontsize=9, color='r', alpha=0.8)

    # Set titles and labels
    ax1.set_title('Individual Weibull Distributions with Peak Lines')
    ax1.set_ylabel('PDF')
    ax1.legend()

    ax2.set_title('Combined Distribution (Sum of Two Weibull Distributions)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('PDF')
    ax2.legend()

    # Set x-axis ticks with 0.1 interval but labels every 0.5 for both plots
    x_max = 8.0  # Fixed maximum range
    x_ticks = np.arange(0, x_max + 0.1, 0.1)
    x_tick_labels = [f'{x:.1f}' if x % 0.5 == 0 else '' for x in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_tick_labels)

    ax1.grid(True)
    ax2.grid(True)
    fig.canvas.draw_idle()


def on_val_x1_change(text):
    """Callback for value input of peak x1"""
    global current_peak_x1
    try:
        val = float(text)
        if peak_x1_min <= val <= peak_x1_max:
            current_peak_x1 = val
            val_x1_box.set_val(f"{val:.2f}")
            update_plot(current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight, update_boxes=False)
    except ValueError:
        pass

def on_val_y1_change(text):
    """Callback for value input of peak y1"""
    global current_peak_y1
    try:
        val = float(text)
        if peak_y1_min <= val <= peak_y1_max:
            current_peak_y1 = val
            val_y1_box.set_val(f"{val:.2f}")
            update_plot(current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight, update_boxes=False)
    except ValueError:
        pass

def on_val_x2_change(text):
    """Callback for value input of peak x2"""
    global current_peak_x2
    try:
        val = float(text)
        if peak_x2_min <= val <= peak_x2_max:
            current_peak_x2 = val
            val_x2_box.set_val(f"{val:.2f}")
            update_plot(current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight, update_boxes=False)
    except ValueError:
        pass

def on_val_y2_change(text):
    """Callback for value input of peak y2"""
    global current_peak_y2
    try:
        val = float(text)
        if peak_y2_min <= val <= peak_y2_max:
            current_peak_y2 = val
            val_y2_box.set_val(f"{val:.2f}")
            update_plot(current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight, update_boxes=False)
    except ValueError:
        pass

def on_val_weight_change(text):
    """Callback for value input of weight"""
    global current_weight
    try:
        val = float(text)
        if 0.0 <= val <= 1.0:
            current_weight = val
            val_weight_box.set_val(f"{val:.2f}")
            update_plot(current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight, update_boxes=False)
    except ValueError:
        pass

def export_data(event):
    """Export current data to CSV for both distributions and weighted sum"""
    peak_x1 = current_peak_x1
    peak_y1 = current_peak_y1
    peak_x2 = current_peak_x2
    peak_y2 = current_peak_y2
    weight_k1 = current_weight
    weight_k2 = 1.0 - weight_k1

    # Find parameters
    k1, lam1 = find_params_for_peak(peak_x1, peak_y1)
    k2, lam2 = find_params_for_peak(peak_x2, peak_y2)

    x_max = 8.0  # Fixed maximum range, same as in update_plot
    x = np.arange(0, x_max + 0.1, 0.1)
    pdf1 = scipy.stats.weibull_min.pdf(x, c=k1, scale=lam1)
    pdf2 = scipy.stats.weibull_min.pdf(x, c=k2, scale=lam2)
    pdf_weighted_sum = weight_k1 * pdf1 + weight_k2 * pdf2

    os.makedirs('output', exist_ok=True)
    filename = 'output/weibull_overlay_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf1', 'pdf2', 'pdf_weighted_sum', 'weight_k1', 'weight_k2'])
        for xi, pdf1i, pdf2i, wsumi in zip(x, pdf1, pdf2, pdf_weighted_sum):
            writer.writerow([xi, pdf1i, pdf2i, wsumi, weight_k1, weight_k2])
    print(f"Data exported to {filename}")

def update_ranges():
    """Update ranges based on text box values"""
    global peak_x1_min, peak_x1_max, peak_y1_min, peak_y1_max
    global peak_x2_min, peak_x2_max, peak_y2_min, peak_y2_max
    try:
        new_peak_x1_min = float(peak_x1_min_box.text)
        new_peak_x1_max = float(peak_x1_max_box.text)
        new_peak_y1_min = float(peak_y1_min_box.text)
        new_peak_y1_max = float(peak_y1_max_box.text)
        new_peak_x2_min = float(peak_x2_min_box.text)
        new_peak_x2_max = float(peak_x2_max_box.text)
        new_peak_y2_min = float(peak_y2_min_box.text)
        new_peak_y2_max = float(peak_y2_max_box.text)

        if (new_peak_x1_min >= new_peak_x1_max or new_peak_x1_min <= 0 or
            new_peak_y1_min >= new_peak_y1_max or new_peak_y1_min <= 0 or
            new_peak_x2_min >= new_peak_x2_max or new_peak_x2_min <= 0 or
            new_peak_y2_min >= new_peak_y2_max or new_peak_y2_min <= 0):
            return  # Invalid ranges

        # Update global ranges
        peak_x1_min = new_peak_x1_min
        peak_x1_max = new_peak_x1_max
        peak_y1_min = new_peak_y1_min
        peak_y1_max = new_peak_y1_max
        peak_x2_min = new_peak_x2_min
        peak_x2_max = new_peak_x2_max
        peak_y2_min = new_peak_y2_min
        peak_y2_max = new_peak_y2_max

        fig.canvas.draw_idle()
    except ValueError:
        pass  # Invalid input, ignore

def on_range_change(text):
    """Callback for range text boxes"""
    update_ranges()

def main():
    global fig, ax1, ax2
    global peak_x1_min_box, peak_x1_max_box, peak_y1_min_box, peak_y1_max_box
    global peak_x2_min_box, peak_x2_max_box, peak_y2_min_box, peak_y2_max_box
    global val_x1_box, val_y1_box, val_x2_box, val_y2_box, val_weight_box
    global current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight
    global peak_x1_min, peak_x1_max, peak_y1_min, peak_y1_max
    global peak_x2_min, peak_x2_max, peak_y2_min, peak_y2_max

    # Create figure with two subplots and space below for controls (1920x1080 optimized)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19.2, 10.8))
    plt.subplots_adjust(bottom=0.45, hspace=0.3)  # Leave more space at bottom for controls and space between subplots

    # Initial parameters and ranges for both distributions
    # Distribution 1: peak at (2, 0.5)
    peak_x1_init = 2.0
    peak_x1_min_init, peak_x1_max_init = 0.1, 5.0
    peak_y1_init = 0.5
    peak_y1_min_init, peak_y1_max_init = 0.1, 2.0

    # Distribution 2: peak at (1.5, 0.6)
    peak_x2_init = 1.5
    peak_x2_min_init, peak_x2_max_init = 0.1, 5.0
    peak_y2_init = 0.6
    peak_y2_min_init, peak_y2_max_init = 0.1, 2.0

    # Initial weight parameter
    weight_init = 0.5

    # Set global current values
    current_peak_x1 = peak_x1_init
    current_peak_y1 = peak_y1_init
    current_peak_x2 = peak_x2_init
    current_peak_y2 = peak_y2_init
    current_weight = weight_init

    # Set global ranges
    peak_x1_min = peak_x1_min_init
    peak_x1_max = peak_x1_max_init
    peak_y1_min = peak_y1_min_init
    peak_y1_max = peak_y1_max_init
    peak_x2_min = peak_x2_min_init
    peak_x2_max = peak_x2_max_init
    peak_y2_min = peak_y2_min_init
    peak_y2_max = peak_y2_max_init

    # Initial plot
    update_plot(current_peak_x1, current_peak_y1, current_peak_x2, current_peak_y2, current_weight)


    # Create value input boxes
    ax_val_x1 = plt.axes([0.1, 0.33, 0.35, 0.03])
    val_x1_box = widgets.TextBox(ax_val_x1, 'Dist1 Peak X:', initial=f"{peak_x1_init:.2f}")

    ax_val_y1 = plt.axes([0.1, 0.28, 0.35, 0.03])
    val_y1_box = widgets.TextBox(ax_val_y1, 'Dist1 Peak Y:', initial=f"{peak_y1_init:.2f}")

    ax_val_x2 = plt.axes([0.55, 0.33, 0.35, 0.03])
    val_x2_box = widgets.TextBox(ax_val_x2, 'Dist2 Peak X:', initial=f"{peak_x2_init:.2f}")

    ax_val_y2 = plt.axes([0.55, 0.28, 0.35, 0.03])
    val_y2_box = widgets.TextBox(ax_val_y2, 'Dist2 Peak Y:', initial=f"{peak_y2_init:.2f}")

    ax_val_weight = plt.axes([0.3, 0.15, 0.4, 0.03])
    val_weight_box = widgets.TextBox(ax_val_weight, 'Weight k1 (k2=1-k1):', initial=f"{weight_init:.2f}")


    # Connect value text boxes
    val_x1_box.on_submit(on_val_x1_change)
    val_y1_box.on_submit(on_val_y1_change)
    val_x2_box.on_submit(on_val_x2_change)
    val_y2_box.on_submit(on_val_y2_change)
    val_weight_box.on_submit(on_val_weight_change)

    # Create range input boxes for distribution 1 (left side)
    ax_peak_x1_min = plt.axes([0.1, 0.38, 0.12, 0.03])
    ax_peak_x1_max = plt.axes([0.25, 0.38, 0.12, 0.03])
    ax_peak_y1_min = plt.axes([0.1, 0.18, 0.12, 0.03])
    ax_peak_y1_max = plt.axes([0.25, 0.18, 0.12, 0.03])

    peak_x1_min_box = widgets.TextBox(ax_peak_x1_min, 'PeakX1 min:', initial=str(peak_x1_min_init))
    peak_x1_max_box = widgets.TextBox(ax_peak_x1_max, 'PeakX1 max:', initial=str(peak_x1_max_init))
    peak_y1_min_box = widgets.TextBox(ax_peak_y1_min, 'PeakY1 min:', initial=str(peak_y1_min_init))
    peak_y1_max_box = widgets.TextBox(ax_peak_y1_max, 'PeakY1 max:', initial=str(peak_y1_max_init))

    # Create range input boxes for distribution 2 (right side)
    ax_peak_x2_min = plt.axes([0.55, 0.38, 0.12, 0.03])
    ax_peak_x2_max = plt.axes([0.7, 0.38, 0.12, 0.03])
    ax_peak_y2_min = plt.axes([0.55, 0.18, 0.12, 0.03])
    ax_peak_y2_max = plt.axes([0.7, 0.18, 0.12, 0.03])

    peak_x2_min_box = widgets.TextBox(ax_peak_x2_min, 'PeakX2 min:', initial=str(peak_x2_min_init))
    peak_x2_max_box = widgets.TextBox(ax_peak_x2_max, 'PeakX2 max:', initial=str(peak_x2_max_init))
    peak_y2_min_box = widgets.TextBox(ax_peak_y2_min, 'PeakY2 min:', initial=str(peak_y2_min_init))
    peak_y2_max_box = widgets.TextBox(ax_peak_y2_max, 'PeakY2 max:', initial=str(peak_y2_max_init))

    # Connect text boxes to range update function
    peak_x1_min_box.on_submit(on_range_change)
    peak_x1_max_box.on_submit(on_range_change)
    peak_y1_min_box.on_submit(on_range_change)
    peak_y1_max_box.on_submit(on_range_change)
    peak_x2_min_box.on_submit(on_range_change)
    peak_x2_max_box.on_submit(on_range_change)
    peak_y2_min_box.on_submit(on_range_change)
    peak_y2_max_box.on_submit(on_range_change)

    # Create export button
    ax_button = plt.axes([0.85, 0.1, 0.12, 0.05])
    export_button = widgets.Button(ax_button, 'Export to CSV')
    export_button.on_clicked(export_data)

    # Add instructions
    fig.text(0.1, 0.02, 'Top: Individual distributions with peaks. Bottom: Weighted sum (k1*Dist1 + k2*Dist2, k1+k2=1). Blue: Dist1, Red: Dist2, Purple: Weighted sum. Parameters k,λ computed automatically from peak coordinates.',
              fontsize=10, style='italic')

    plt.show()

if __name__ == "__main__":
    main()