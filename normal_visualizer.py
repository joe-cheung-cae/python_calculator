import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import csv
import os

def update_plot(mu, sigma):
    """Update the plot with new parameters"""
    if sigma <= 0:
        return  # Skip invalid parameters

    ax.clear()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
    ax.plot(x, pdf)
    ax.set_title(f'Normal Distribution PDF (μ={mu:.3f}, σ={sigma:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('PDF')
    ax.grid(True)
    fig.canvas.draw_idle()

def on_mu_change(val):
    """Callback for mean parameter slider"""
    update_plot(val, sigma_slider.val)

def on_sigma_change(val):
    """Callback for standard deviation parameter slider"""
    update_plot(mu_slider.val, val)

def export_data(event):
    """Export current data to CSV"""
    mu = mu_slider.val
    sigma = sigma_slider.val
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)

    os.makedirs('output', exist_ok=True)
    filename = 'output/normal_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf'])
        for xi, pdfi in zip(x, pdf):
            writer.writerow([xi, pdfi])
    print(f"Data exported to {filename}")

def update_slider_ranges():
    """Update slider ranges based on text box values"""
    try:
        mu_min = float(mu_min_box.text)
        mu_max = float(mu_max_box.text)
        sigma_min = float(sigma_min_box.text)
        sigma_max = float(sigma_max_box.text)

        if mu_min >= mu_max or sigma_min >= sigma_max or sigma_min <= 0:
            return  # Invalid ranges

        # Update mu slider
        mu_slider.valmin = mu_min
        mu_slider.valmax = mu_max
        mu_slider.ax.set_xlim(mu_min, mu_max)
        if mu_slider.val < mu_min:
            mu_slider.set_val(mu_min)
        elif mu_slider.val > mu_max:
            mu_slider.set_val(mu_max)

        # Update sigma slider
        sigma_slider.valmin = sigma_min
        sigma_slider.valmax = sigma_max
        sigma_slider.ax.set_xlim(sigma_min, sigma_max)
        if sigma_slider.val < sigma_min:
            sigma_slider.set_val(sigma_min)
        elif sigma_slider.val > sigma_max:
            sigma_slider.set_val(sigma_max)

        fig.canvas.draw_idle()
    except ValueError:
        pass  # Invalid input, ignore

def on_range_change(text):
    """Callback for range text boxes"""
    update_slider_ranges()

def main():
    global fig, ax, mu_slider, sigma_slider, mu_min_box, mu_max_box, sigma_min_box, sigma_max_box

    # Create figure with single plot and space below for controls
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.35)  # Leave more space at bottom for controls

    # Initial parameters and ranges
    mu_init = 0.0
    sigma_init = 1.0
    mu_min_init, mu_max_init = -5.0, 5.0
    sigma_min_init, sigma_max_init = 0.1, 5.0

    # Initial plot
    update_plot(mu_init, sigma_init)

    # Create sliders
    ax_mu = plt.axes([0.1, 0.2, 0.35, 0.03])
    ax_sigma = plt.axes([0.1, 0.15, 0.35, 0.03])

    mu_slider = widgets.Slider(ax_mu, 'Mean (μ)', mu_min_init, mu_max_init, valinit=mu_init, valstep=0.1)
    sigma_slider = widgets.Slider(ax_sigma, 'Std Dev (σ)', sigma_min_init, sigma_max_init, valinit=sigma_init, valstep=0.1)

    # Connect sliders to update function
    mu_slider.on_changed(on_mu_change)
    sigma_slider.on_changed(on_sigma_change)

    # Create range input boxes with better spacing
    ax_mu_min = plt.axes([0.1, 0.25, 0.12, 0.03])
    ax_mu_max = plt.axes([0.25, 0.25, 0.12, 0.03])
    ax_sigma_min = plt.axes([0.1, 0.08, 0.12, 0.03])
    ax_sigma_max = plt.axes([0.25, 0.08, 0.12, 0.03])

    mu_min_box = widgets.TextBox(ax_mu_min, 'μ min:', initial=str(mu_min_init))
    mu_max_box = widgets.TextBox(ax_mu_max, 'μ max:', initial=str(mu_max_init))
    sigma_min_box = widgets.TextBox(ax_sigma_min, 'σ min:', initial=str(sigma_min_init))
    sigma_max_box = widgets.TextBox(ax_sigma_max, 'σ max:', initial=str(sigma_max_init))

    # Connect text boxes to range update function
    mu_min_box.on_submit(on_range_change)
    mu_max_box.on_submit(on_range_change)
    sigma_min_box.on_submit(on_range_change)
    sigma_max_box.on_submit(on_range_change)

    # Create export button
    ax_button = plt.axes([0.6, 0.1, 0.2, 0.05])
    export_button = widgets.Button(ax_button, 'Export to CSV')
    export_button.on_clicked(export_data)

    # Add instructions
    fig.text(0.1, 0.02, 'Adjust range values and use sliders to control Normal distribution parameters. Click "Export to CSV" to save current data.',
              fontsize=10, style='italic')

    plt.show()

if __name__ == "__main__":
    main()