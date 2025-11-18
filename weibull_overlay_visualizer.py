import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import csv
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


def update_plot(peak_x1, peak_y1, peak_x2, peak_y2, weight_k1=0.5):
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

def on_peak_x1_change(val):
    """Callback for peak x slider of distribution 1"""
    update_plot(val, peak_y1_slider.val, peak_x2_slider.val, peak_y2_slider.val, weight_slider.val)

def on_peak_y1_change(val):
    """Callback for peak y slider of distribution 1"""
    update_plot(peak_x1_slider.val, val, peak_x2_slider.val, peak_y2_slider.val, weight_slider.val)

def on_peak_x2_change(val):
    """Callback for peak x slider of distribution 2"""
    update_plot(peak_x1_slider.val, peak_y1_slider.val, val, peak_y2_slider.val, weight_slider.val)

def on_peak_y2_change(val):
    """Callback for peak y slider of distribution 2"""
    update_plot(peak_x1_slider.val, peak_y1_slider.val, peak_x2_slider.val, val, weight_slider.val)

def on_weight_change(val):
    """Callback for weight parameter slider"""
    update_plot(peak_x1_slider.val, peak_y1_slider.val, peak_x2_slider.val, peak_y2_slider.val, val)

def export_data(event):
    """Export current data to CSV for both distributions and weighted sum"""
    peak_x1 = peak_x1_slider.val
    peak_y1 = peak_y1_slider.val
    peak_x2 = peak_x2_slider.val
    peak_y2 = peak_y2_slider.val
    weight_k1 = weight_slider.val
    weight_k2 = 1.0 - weight_k1

    # Find parameters
    k1, lam1 = find_params_for_peak(peak_x1, peak_y1)
    k2, lam2 = find_params_for_peak(peak_x2, peak_y2)

    x_max = 8.0  # Fixed maximum range, same as in update_plot
    x = np.arange(0, x_max + 0.1, 0.1)
    pdf1 = scipy.stats.weibull_min.pdf(x, c=k1, scale=lam1)
    pdf2 = scipy.stats.weibull_min.pdf(x, c=k2, scale=lam2)
    pdf_weighted_sum = weight_k1 * pdf1 + weight_k2 * pdf2

    filename = 'weibull_overlay_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf1', 'pdf2', 'pdf_weighted_sum', 'weight_k1', 'weight_k2'])
        for xi, pdf1i, pdf2i, wsumi in zip(x, pdf1, pdf2, pdf_weighted_sum):
            writer.writerow([xi, pdf1i, pdf2i, wsumi, weight_k1, weight_k2])
    print(f"Data exported to {filename}")

def update_slider_ranges():
    """Update slider ranges based on text box values"""
    try:
        peak_x1_min = float(peak_x1_min_box.text)
        peak_x1_max = float(peak_x1_max_box.text)
        peak_y1_min = float(peak_y1_min_box.text)
        peak_y1_max = float(peak_y1_max_box.text)
        peak_x2_min = float(peak_x2_min_box.text)
        peak_x2_max = float(peak_x2_max_box.text)
        peak_y2_min = float(peak_y2_min_box.text)
        peak_y2_max = float(peak_y2_max_box.text)

        if (peak_x1_min >= peak_x1_max or peak_x1_min <= 0 or
            peak_y1_min >= peak_y1_max or peak_y1_min <= 0 or
            peak_x2_min >= peak_x2_max or peak_x2_min <= 0 or
            peak_y2_min >= peak_y2_max or peak_y2_min <= 0):
            return  # Invalid ranges

        # Update distribution 1 sliders
        peak_x1_slider.valmin = peak_x1_min
        peak_x1_slider.valmax = peak_x1_max
        peak_x1_slider.ax.set_xlim(peak_x1_min, peak_x1_max)
        if peak_x1_slider.val < peak_x1_min:
            peak_x1_slider.set_val(peak_x1_min)
        elif peak_x1_slider.val > peak_x1_max:
            peak_x1_slider.set_val(peak_x1_max)

        peak_y1_slider.valmin = peak_y1_min
        peak_y1_slider.valmax = peak_y1_max
        peak_y1_slider.ax.set_xlim(peak_y1_min, peak_y1_max)
        if peak_y1_slider.val < peak_y1_min:
            peak_y1_slider.set_val(peak_y1_min)
        elif peak_y1_slider.val > peak_y1_max:
            peak_y1_slider.set_val(peak_y1_max)

        # Update distribution 2 sliders
        peak_x2_slider.valmin = peak_x2_min
        peak_x2_slider.valmax = peak_x2_max
        peak_x2_slider.ax.set_xlim(peak_x2_min, peak_x2_max)
        if peak_x2_slider.val < peak_x2_min:
            peak_x2_slider.set_val(peak_x2_min)
        elif peak_x2_slider.val > peak_x2_max:
            peak_x2_slider.set_val(peak_x2_max)

        peak_y2_slider.valmin = peak_y2_min
        peak_y2_slider.valmax = peak_y2_max
        peak_y2_slider.ax.set_xlim(peak_y2_min, peak_y2_max)
        if peak_y2_slider.val < peak_y2_min:
            peak_y2_slider.set_val(peak_y2_min)
        elif peak_y2_slider.val > peak_y2_max:
            peak_y2_slider.set_val(peak_y2_max)

        fig.canvas.draw_idle()
    except ValueError:
        pass  # Invalid input, ignore

def on_range_change(text):
    """Callback for range text boxes"""
    update_slider_ranges()

def main():
    global fig, ax1, ax2, peak_x1_slider, peak_y1_slider, peak_x2_slider, peak_y2_slider, weight_slider
    global peak_x1_min_box, peak_x1_max_box, peak_y1_min_box, peak_y1_max_box
    global peak_x2_min_box, peak_x2_max_box, peak_y2_min_box, peak_y2_max_box

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

    # Initial plot
    update_plot(peak_x1_init, peak_y1_init, peak_x2_init, peak_y2_init, weight_init)

    # Create sliders for distribution 1 (left side)
    ax_peak_x1 = plt.axes([0.1, 0.33, 0.35, 0.03])
    peak_x1_slider = widgets.Slider(ax_peak_x1, 'Dist1 Peak X', peak_x1_min_init, peak_x1_max_init, valinit=peak_x1_init, valstep=0.1)

    ax_peak_y1 = plt.axes([0.1, 0.28, 0.35, 0.03])
    peak_y1_slider = widgets.Slider(ax_peak_y1, 'Dist1 Peak Y', peak_y1_min_init, peak_y1_max_init, valinit=peak_y1_init, valstep=0.01)

    # Create sliders for distribution 2 (right side)
    ax_peak_x2 = plt.axes([0.55, 0.33, 0.35, 0.03])
    peak_x2_slider = widgets.Slider(ax_peak_x2, 'Dist2 Peak X', peak_x2_min_init, peak_x2_max_init, valinit=peak_x2_init, valstep=0.1)

    ax_peak_y2 = plt.axes([0.55, 0.28, 0.35, 0.03])
    peak_y2_slider = widgets.Slider(ax_peak_y2, 'Dist2 Peak Y', peak_y2_min_init, peak_y2_max_init, valinit=peak_y2_init, valstep=0.01)

    # Create weight slider (center bottom)
    ax_weight = plt.axes([0.3, 0.15, 0.4, 0.03])
    weight_slider = widgets.Slider(ax_weight, 'Weight k1 (k2=1-k1)', 0.0, 1.0, valinit=weight_init, valstep=0.01)

    # Connect sliders to update function
    peak_x1_slider.on_changed(on_peak_x1_change)
    peak_y1_slider.on_changed(on_peak_y1_change)
    peak_x2_slider.on_changed(on_peak_x2_change)
    peak_y2_slider.on_changed(on_peak_y2_change)
    weight_slider.on_changed(on_weight_change)

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