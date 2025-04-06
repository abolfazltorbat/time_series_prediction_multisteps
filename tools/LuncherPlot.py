import time
import os
import logging
import requests
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Existing imports from your code
# ------------------------------------------------------------------------------
from tools.PlotPrediction import plot_future_prediction, RealTimePlotter
# from tools.FetchData import fetch_data  # If needed for anything else

# -------------------------------------------------------------------------
# 1) This function will fetch data from your previously defined API endpoints
#    (e.g., /get_data, /get_prediction, /get_prediction_time) every 10 seconds.
#    Then it will update a plot with the new data.
# -------------------------------------------------------------------------
def real_time_prediction_loop(
    api_base_url: str = "http://localhost:5500",
    update_interval: int = 10
):
    """
    Periodically fetch the latest predictions, new_data, and date_time from
    your API endpoints and then plot them. The latest prediction update time
    will be included in the plot title.
    """

    # Initialize a real-time plotter (from your existing code).
    # If you'd rather use plot_future_prediction directly, adapt accordingly.
    plotter = RealTimePlotter(horizon=90)

    # Turn interactive mode on, so the plot updates without blocking.
    plt.ion()

    while True:
        try:
            # ---------------------------------------------------------------
            # 1) Fetch new_data and date_time
            # ---------------------------------------------------------------
            data_resp = requests.get(f"{api_base_url}/get_data")
            new_data, date_time = None, None
            if data_resp.status_code == 200:
                data_json = data_resp.json()
                new_data = data_json.get("new_data", None)
                date_time = data_json.get("date_time", None)

            # ---------------------------------------------------------------
            # 2) Fetch predictions
            # ---------------------------------------------------------------
            pred_resp = requests.get(f"{api_base_url}/get_prediction")
            predictions = None
            if pred_resp.status_code == 200:
                pred_json = pred_resp.json()
                predictions = pred_json.get("predictions", None)

            # ---------------------------------------------------------------
            # 3) Fetch the latest update time for the predictions
            # ---------------------------------------------------------------
            time_resp = requests.get(f"{api_base_url}/get_prediction_time")
            latest_pred_update_time = None
            if time_resp.status_code == 200:
                time_json = time_resp.json()
                latest_pred_update_time = time_json.get("latest_prediction_update_time", None)

            # ---------------------------------------------------------------
            # 4) Update the plot if data is available
            # ---------------------------------------------------------------

            if new_data is not None and predictions is not None:

                # Update the real-time plot
                plotter.update_plot(new_data, predictions, date_time,horizon=90,latest_update=latest_pred_update_time)

                # Let matplotlib process UI events
                plt.pause(0.1)

        except Exception as e:
            logging.error(f"Error fetching or plotting data: {e}")

        # ---------------------------------------------------------------
        # 5) Wait 10 seconds before fetching again
        # ---------------------------------------------------------------
        # time.sleep(update_interval)
        start_time = time.time()
        while time.time() - start_time < update_interval:
            plt.pause(0.1)

# -------------------------------------------------------------------------
# 2) Main entry point
# -------------------------------------------------------------------------
def main():
    """
    Starts the real-time plotting loop. This is a blocking loop that repeatedly
    fetches from the API and updates the plot.
    """
    print('in main')
    real_time_prediction_loop(
        api_base_url="http://localhost:5500",  # Adjust if your API is elsewhere
        update_interval=10
    )

if __name__ == "__main__":
    main()
