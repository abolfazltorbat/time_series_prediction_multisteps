from matplotlib.ticker import AutoMinorLocator
# plt.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os
class RealTimePlotter:
    def __init__(self,horizon=None):
        # Use a dark style as the base
        plt.style.use('dark_background')

        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=80)
        self.setup_plot_style()

        # Change the figure name
        self.fig.canvas.manager.set_window_title(f"Future Forecast, Horizon: {horizon}")
        # icon_path = "D:\\Academic Works\\work-pc\\academic\\TimeSeriesPrediction\\time series prediction\\pythonProject1\\tools\\icon.png"
        # icon = Image.open(icon_path)
        # icon = ImageTk.PhotoImage(icon)
        # window = plt.get_current_fig_manager().window
        # window.tk.call('wm', 'iconphoto', window._w, icon)

        # Enable toolbar for zoom only
        self.toolbar = plt.get_current_fig_manager().toolbar
        self.remove_toolbar_buttons_except_zoom()

        # Add event handlers for mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_event)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_event)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_event)

    def remove_toolbar_buttons_except_zoom(self):
        """Remove all toolbar buttons except zoom and home."""
        if hasattr(self.toolbar, '_buttons'):
            for button in list(self.toolbar._buttons.keys()):
                if 'zoom' not in str(button).lower() and 'home' not in str(button).lower():
                    self.toolbar._buttons[button].pack_forget()

    def on_mouse_event(self, event):
        """Handle mouse events to keep the plot responsive"""
        if self.toolbar.mode:  # If we're in zoom mode
            self.fig.canvas.draw_idle()


    def setup_plot_style(self,
                         x_tick_fontsize=8,
                         y_tick_fontsize=10,
                         use_minor_xticks=True,
                         minor_xticks_interval=2):
        """
        Set up the initial plot style to look more like a trading chart:
        - Dark background
        - Spines and grid lines in subtle shades
        - Ticks and labels in lighter shades
        - Optional minor x ticks for higher resolution
        - Optional font size configuration for ticks

        Args:
            x_tick_fontsize (int)       : Font size for x-axis tick labels
            y_tick_fontsize (int)       : Font size for y-axis tick labels
            use_minor_xticks (bool)     : Whether to enable minor ticks on x-axis
            minor_xticks_interval (int) : How many subdivisions for minor ticks
                                          (used by AutoMinorLocator)
        """
        self.ax.set_facecolor('#1E1E1E')

        # Move y-axis to the right
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")

        # Set axis labels
        self.ax.set_ylabel('Price', fontsize=12, fontweight='bold', color='white')

        # Style the grid
        self.ax.grid(True, color='#444444', linestyle='--', alpha=0.7)

        # Make the border (spines) narrower and color them subtly
        for spine in self.ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color('#888888')

        # Major tick parameters
        self.ax.tick_params(axis='x', which='major', colors='white', labelsize=x_tick_fontsize)
        self.ax.tick_params(axis='y', which='major', colors='white', labelsize=y_tick_fontsize)

        # Optionally enable minor x ticks for finer resolution
        if use_minor_xticks:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(minor_xticks_interval))
            self.ax.tick_params(axis='x', which='minor', colors='white', labelsize=x_tick_fontsize)
            # If you also want grid lines for minor ticks:
            self.ax.grid(True, which='minor', color='#444444', linestyle=':', alpha=0.4)

        # Make sure layout fits
        plt.tight_layout()

    def update_plot(self,
                    original_data,
                    predictions,
                    date_time=None,
                    # --- New style parameters for historical vs. predictions ---
                    hist_color='#00FF00',
                    hist_linestyle='-',
                    hist_linewidth=1.5,
                    pred_color='red',
                    pred_line_color = '#FF4500',
                    pred_linestyle=':', #  '-', '--', '-.', ':'
                    pred_linewidth=1.5,
                    horizon = None,
                    latest_update = None):
        """
        Update the plot with new data (historical vs. future predictions).

        Args:
            original_data (array-like): Historical time series data.
            predictions (array-like): Future predicted values.
            date_time (pd.DatetimeIndex or None): Corresponding datetime index for original_data.
            hist_color, hist_linestyle, hist_linewidth : Style attributes for historical data line.
            pred_color, pred_linestyle, pred_linewidth : Style attributes for prediction line.
        """
        self.fig.canvas.manager.set_window_title(f"Future Forecast, Horizon: {horizon}, latest update: {latest_update}")

        self.ax.clear()
        self.setup_plot_style()

        # If we have actual datetime information
        if date_time is not None:
            # Convert to pandas datetime index if needed
            if not isinstance(date_time, pd.DatetimeIndex):
                try:
                    date_time = pd.to_datetime(date_time, format='%Y-%m-%dT%H:%M:%S')
                except ValueError:
                    # Fallback to general parsing if specific format fails
                    date_time = pd.to_datetime(date_time)
            elif isinstance(date_time, (list, tuple)):
                # Convert list of strings to pandas datetime
                date_time = pd.to_datetime(date_time, format='%Y-%m-%dT%H:%M:%S')

            last_date = date_time[-1]
            # Try to infer frequency; if none found, approximate with time deltas
            freq = pd.infer_freq(date_time)
            if freq is None and len(date_time) >= 2:
                # Calculate all time differences
                time_diffs = date_time[1:] - date_time[:-1]

                # Find the most common time difference
                unique_diffs, counts = np.unique(time_diffs, return_counts=True)
                most_common_diff = unique_diffs[np.argmax(counts)]

                freq = most_common_diff
            elif freq is None:
                # fallback if we have only a single point
                freq = 'D'  # or any default

            if isinstance(freq, str) and freq == 'min' or freq==60000000000:
                # Use 'min' instead of 'T' for minutes
                formatted_freq = '1min'  # Updated from '1T' to '1min'
            elif isinstance(freq, pd.Timedelta):
                # If freq is a Timedelta, convert to appropriate string format
                minutes = freq.total_seconds() / 60
                formatted_freq = f'{int(minutes)}min'  # Updated from 'T' to 'min'
            else:
                # Keep original frequency for other cases
                formatted_freq = freq

            # Generate future dates using the formatted frequency
            future_dates = pd.date_range(
                start=last_date,
                periods=len(predictions),
                freq=formatted_freq
            )

            # Plot historical data
            self.ax.plot(date_time,
                         original_data,
                         label='Historical Data',
                         color=hist_color,
                         linestyle=hist_linestyle,
                         linewidth=hist_linewidth)

            # Plot future predictions
            self.ax.plot(future_dates,
                         predictions,
                         label='Future Predictions',
                         color=pred_color,
                         linestyle=pred_linestyle,
                         linewidth=pred_linewidth)

            # --- Connect last historical point to first prediction point ---
            self.ax.plot([last_date, future_dates[0]],
                         [original_data[-1], predictions[0]],
                         color=pred_color,
                         linestyle=pred_linestyle,
                         linewidth=pred_linewidth)

            plt.xticks(rotation=45)
            self.ax.set_xlabel('Time', fontsize=12, fontweight='bold', color='white')

        else:
            # No datetime => just use numeric indices
            time_historical = np.arange(len(original_data))
            time_future = np.arange(len(original_data), len(original_data) + len(predictions))

            # Plot historical data
            self.ax.plot(time_historical,
                         original_data,
                         label='Historical Data',
                         color=hist_color,
                         linestyle=hist_linestyle,
                         linewidth=hist_linewidth)

            # Plot future predictions
            self.ax.plot(time_future,
                         predictions,
                         label='Future Predictions',
                         color=pred_color,
                         linestyle=pred_linestyle,
                         linewidth=pred_linewidth)

            # --- Connect last historical point to first prediction point ---
            self.ax.plot([time_historical[-1], time_future[0]],
                         [original_data[-1], predictions[0]],
                         color=pred_color,
                         linestyle=pred_linestyle,
                         linewidth=pred_linewidth)

            self.ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold', color='white')

        # plot the legend
        legend = self.ax.legend(loc='upper left', frameon=False)  # frameon=False removes the legend border
        for text in legend.get_texts():
            text.set_color('white')  # Make legend text white
            text.set_fontweight('bold')  # Optionally make it bold (for 'sharp' text)

        # -----------------------------------------------------------------------
        # --- Add dashed horizontal lines for the latest and predicted prices ---
        latest_price = original_data[-1]
        latest_pred_price = predictions[-1]
        # Horizontal line for the latest price (red)
        self.ax.axhline(y=latest_price,
                        color='green',
                        linestyle=':',
                        label='Latest Price')

        # Horizontal line for the latest predicted price (light red)
        self.ax.axhline(y=latest_pred_price,
                        color='red',  # pick any "light red" hex
                        linestyle=':',
                        label='Latest Predicted Price')

        # --- Add labels on the y-axis for the horizontal lines ---
        # Retrieve current y-ticks
        current_yticks = self.ax.get_yticks().tolist()

        # Add latest_price and latest_pred_price to y-ticks if not already present
        if latest_price not in current_yticks:
            current_yticks.append(latest_price)
        if latest_pred_price not in current_yticks:
            current_yticks.append(latest_pred_price)

        # Sort the y-ticks
        current_yticks = sorted(current_yticks)

        # Set the updated y-ticks
        self.ax.set_yticks(current_yticks)

        # Set y-tick labels, highlighting the latest prices
        yticks_labels = []
        for tick in current_yticks:
            if np.isclose(tick, latest_price):
                label = f'{tick:.2f}'
            elif np.isclose(tick, latest_pred_price):
                label = f'{tick:.2f}'
            else:
                label = f'{tick:.2f}'
            yticks_labels.append(label)

        self.ax.set_yticklabels(yticks_labels, color='white', fontweight='bold')


        plt.draw()
        plt.pause(0.1)
def plot_future_prediction(original_data,
                           predictions,
                           params,
                           results_dir,
                           is_in_future_prediction=False,
                           count = '1_',
                           is_plot = True,
                           date_time=None):
    """
    Plot the historical data and future predictions.

    Args:
        original_data: Historical data points
        predictions: Predicted future values
        params: Configuration parameters
        results_dir: Directory to save results
        is_in_future_prediction: Boolean flag for future prediction mode
        date_time: Optional datetime index for x-axis
    """
    plt.style.use('grayscale')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=params['dpi'])

    if date_time is not None:
        # Calculate future dates by extending the time series
        # Assuming same frequency as historical data
        last_date = date_time[-1]
        freq = pd.infer_freq(date_time)
        if freq is None:
            # If frequency cannot be inferred, assume same time delta as last two points
            freq = date_time[-1] - date_time[-2]

        future_dates = pd.date_range(
            start=last_date + freq,
            periods=len(predictions),
            freq=freq
        )

        # Plot with datetime x-axis
        ax.plot(date_time, original_data,
                label='Historical Data',
                color='#2C3E50',
                linewidth=1.5)

        ax.plot(future_dates, predictions,
                label='Future Predictions',
                color='#E74C3C',
                linewidth=1.5,
                linestyle='--')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    else:
        # Original plotting code for numeric time steps
        time_historical = np.arange(len(original_data))
        ax.plot(time_historical, original_data,
                label='Historical Data',
                color='#2C3E50',
                linewidth=1.5)

        time_future = np.arange(len(original_data), len(original_data) + len(predictions))
        ax.plot(time_future, predictions,
                label='Future Predictions',
                color='#E74C3C',
                linewidth=1.5,
                linestyle='--')

        ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')

    # Rest of the customization remains the same
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    # ax.set_title('Time Series Forecasting Results',
    #              fontsize=8,
    #              fontweight='bold',
    #              pad=20)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()

    if is_in_future_prediction and is_plot:
        plt.show()

    if not is_in_future_prediction:
        save_path = os.path.join(results_dir,count + "future_prediction.png")
        plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.1)

    if params.get('is_plot', False) and  is_plot:
        plt.show()

    plt.close()