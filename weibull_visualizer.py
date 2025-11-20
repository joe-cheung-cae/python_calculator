import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import csv
import os

def update_plot(k, lam):
    """Update the plot with new parameters"""
    if k <= 0 or lam <= 0:
        return  # Skip invalid parameters

    ax.clear()
    x = np.linspace(0, lam * 5, 1000)
    pdf = scipy.stats.weibull_min.pdf(x, c=k, scale=lam)
    ax.plot(x, pdf)
    ax.set_title(f'Weibull Distribution PDF (k={k:.3f}, λ={lam:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('PDF')
    ax.grid(True)
    fig.canvas.draw_idle()

def on_k_change(val):
    """Callback for shape parameter slider"""
    update_plot(val, lam_slider.val)

def on_lam_change(val):
    """Callback for scale parameter slider"""
    update_plot(k_slider.val, val)

def export_data(event):
    """Export current data to CSV"""
    k = k_slider.val
    lam = lam_slider.val
    x = np.linspace(0, lam * 5, 1000)
    pdf = scipy.stats.weibull_min.pdf(x, c=k, scale=lam)

    os.makedirs('output', exist_ok=True)
    filename = 'output/weibull_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'pdf'])
        for xi, pdfi in zip(x, pdf):
            writer.writerow([xi, pdfi])
    print(f"Data exported to {filename}")

def update_slider_ranges():
    """Update slider ranges based on text box values"""
    try:
        k_min = float(k_min_box.text)
        k_max = float(k_max_box.text)
        lam_min = float(lam_min_box.text)
        lam_max = float(lam_max_box.text)

        if k_min >= k_max or lam_min >= lam_max or k_min <= 0 or lam_min <= 0:
            return  # Invalid ranges

        # Update k slider
        k_slider.valmin = k_min
        k_slider.valmax = k_max
        k_slider.ax.set_xlim(k_min, k_max)
        if k_slider.val < k_min:
            k_slider.set_val(k_min)
        elif k_slider.val > k_max:
            k_slider.set_val(k_max)

        # Update lambda slider
        lam_slider.valmin = lam_min
        lam_slider.valmax = lam_max
        lam_slider.ax.set_xlim(lam_min, lam_max)
        if lam_slider.val < lam_min:
            lam_slider.set_val(lam_min)
        elif lam_slider.val > lam_max:
            lam_slider.set_val(lam_max)

        fig.canvas.draw_idle()
    except ValueError:
        pass  # Invalid input, ignore

def on_range_change(text):
    """Callback for range text boxes"""
    update_slider_ranges()

def main():
    global fig, ax, k_slider, lam_slider, k_min_box, k_max_box, lam_min_box, lam_max_box

    # Create figure with single plot and space below for controls
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(bottom=0.35)  # Leave more space at bottom for controls

    # Initial parameters and ranges
    k_init = 2.0
    lam_init = 1.414
    k_min_init, k_max_init = 0.1, 5.0
    lam_min_init, lam_max_init = 0.1, 5.0

    # Initial plot
    update_plot(k_init, lam_init)

    # Create sliders
    ax_k = plt.axes([0.1, 0.2, 0.35, 0.03])
    ax_lam = plt.axes([0.1, 0.15, 0.35, 0.03])

    k_slider = widgets.Slider(ax_k, 'Shape (k)', k_min_init, k_max_init, valinit=k_init, valstep=0.1)
    lam_slider = widgets.Slider(ax_lam, 'Scale (λ)', lam_min_init, lam_max_init, valinit=lam_init, valstep=0.1)

    # Connect sliders to update function
    k_slider.on_changed(on_k_change)
    lam_slider.on_changed(on_lam_change)

    # Create range input boxes with better spacing
    ax_k_min = plt.axes([0.1, 0.25, 0.12, 0.03])
    ax_k_max = plt.axes([0.25, 0.25, 0.12, 0.03])
    ax_lam_min = plt.axes([0.1, 0.08, 0.12, 0.03])
    ax_lam_max = plt.axes([0.25, 0.08, 0.12, 0.03])

    k_min_box = widgets.TextBox(ax_k_min, 'k min:', initial=str(k_min_init))
    k_max_box = widgets.TextBox(ax_k_max, 'k max:', initial=str(k_max_init))
    lam_min_box = widgets.TextBox(ax_lam_min, 'λ min:', initial=str(lam_min_init))
    lam_max_box = widgets.TextBox(ax_lam_max, 'λ max:', initial=str(lam_max_init))

    # Connect text boxes to range update function
    k_min_box.on_submit(on_range_change)
    k_max_box.on_submit(on_range_change)
    lam_min_box.on_submit(on_range_change)
    lam_max_box.on_submit(on_range_change)

    # Create export button
    ax_button = plt.axes([0.6, 0.1, 0.2, 0.05])
    export_button = widgets.Button(ax_button, 'Export to CSV')
    export_button.on_clicked(export_data)

    # Add instructions
    fig.text(0.1, 0.02, 'Adjust range values and use sliders to control Weibull parameters. Click "Export to CSV" to save current data.',
             fontsize=10, style='italic')

    plt.show()

if __name__ == "__main__":
    main()