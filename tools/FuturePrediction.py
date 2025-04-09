import logging
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from typing import Optional
import asyncio
import threading
from glob import glob
from pathlib import Path

from fastapi import FastAPI
import uvicorn

from tools.Training import find_model

from tools.PreProcessing import (
    process_window,
    progressive_downsample,
    extract_features,
    de_difference,
    compute_difference,
    impute_missing_values,
    remove_outliers,
    normalize_data
)
from tools.Training import AntiPersistenceLoss
from tools.FetchData import fetch_data
from tools.PlotPrediction import plot_future_prediction  # Import kept as-is

# ---------------------------------------------------------------------
# Global variables to store the latest data, predictions, and timestamps
latest_preds = None  # Will hold the most recent predictions from periodic task
latest_new_data = None
latest_date_time = None
latest_pred_update_time = None
# ---------------------------------------------------------------------
def load_model_and_params(model_path: str, params_path: str):
    """
    Loads the Keras model and parameter dictionary from disk.
    """
    model_path  = find_model(model_path)
    params_path = str(Path(params_path).glob("*_params.pkl").__next__())


    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'AntiPersistenceLoss': AntiPersistenceLoss})

    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    return model, params

def future_prediction(model,
                      data,
                      original_data,
                      params,
                      scaler=None,
                      results_dir=None,
                      is_in_future_prediction=False,
                      verbose="all",
                      is_ModelInterpretability=False,
                      date_time=None):
    """
    Perform future prediction using the model, with adjustable verbosity.
    """
    logger = logging.getLogger('TimeSeriesModel')
    if verbose != "off":
        logger.info("Performing future prediction")

    predictions = []
    window_size = params['window_size']
    num_features = data.shape[1]
    input_seq = data
    horizon = params['horizon']

    start_time = time.time()

    if params.get('prediction_approach', 'recursive')=='multi_steps' and not params.get('multi_steps_recursive', False):
        horizon = 1

    progress_interval = max(1, horizon // 20)

    if verbose == "all":
        print("\n📊 Future Prediction Progress:")
        print("=" * 80)
        print(f"Window Size: {window_size} | Features: {num_features} | Horizon: {horizon}")
        print("=" * 80)

    params['multi_steps_recursive'] = False
    horizon = 1
    for step in range(horizon):

        step_start_time = time.time()
        input_seq = input_seq[-window_size:, :]
        X, last_actual_value, initial_value = process_window(-1, input_seq, params, window_size, num_features)
        if params.get('prediction_approach', 'recursive')=='recursive':
            input_seq_reshaped = X.reshape(1, -1)
            pred = model.predict(input_seq_reshaped, verbose=0)
        else: # multi_steps
            if params.get('multi_steps_recursive', False):
                x_input = X.reshape(1, X.shape[0], 1)
                y_pred = model.predict(x_input, verbose=0)
                pred = y_pred[:, 0].reshape(-1, 1)
            else:
                x_input = X.reshape(1, X.shape[0], 1)
                y_pred = model.predict(x_input, verbose=0)
                pred = y_pred[:, :].reshape(-1, 1)



        if params.get('normalization_type', None) == 'window':
            input_seq, scaler = normalize_data(
                input_seq,
                method=params['normalization_method'],
                logger=logging
            )

        if is_ModelInterpretability:
            from tools.ModelInterpretability import compute_grad_cam, plot_1d_grad_cam
            grad_cam = compute_grad_cam(model, input_seq_reshaped[:, :, np.newaxis], layer_name="conv1d")
            plot_1d_grad_cam(grad_cam)

        if scaler is not None and params.get('normalization_type', None) == 'all':
            pred_denorm = scaler.inverse_transform(pred)
            if params.get('prediction_approach', 'recursive') == 'multi_steps' and not params.get(
                    'multi_steps_recursive', False):
                predictions = pred_denorm
            else:
                predictions.append(pred_denorm[0])
            new_input = scaler.transform(pred_denorm)
        elif params.get('normalization_type', None) == 'window':
            pred_denorm = scaler.inverse_transform(pred)
            if params.get('prediction_approach', 'recursive') == 'multi_steps' and not params.get(
                    'multi_steps_recursive', False):
                predictions = pred_denorm
            else:
                predictions.append(pred_denorm[0])
            new_input = pred
        else:
            if params.get('prediction_approach', 'recursive') == 'multi_steps' and not params.get(
                    'multi_steps_recursive', False):
                predictions = pred
            else:
                predictions.append(pred[0])
            new_input = pred

        input_seq = np.vstack((input_seq[1:], new_input))

        if params.get('normalization_type', None) == 'window':
            input_seq = scaler.inverse_transform(input_seq)

        if (step % progress_interval == 0 or step == horizon - 1) and verbose in ["all", "short"]:
            elapsed_time = time.time() - start_time
            step_time = time.time() - step_start_time
            progress = (step + 1) / horizon
            eta = (elapsed_time / (step + 1)) * (horizon - step - 1)

            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            print(
                f"\r[{bar}] {progress * 100:.1f}% | "
                f"Step: {step + 1}/{horizon} | "
                f"Elapsed: {elapsed_time:.1f}s | "
                f"ETA: {eta:.1f}s | "
                f"Step Time: {step_time:.3f}s",
                end=''
            )
            if step == horizon - 1:
                if verbose == "all":
                    print("\n" + "=" * 80)
                    print(f"✅ Prediction completed in {elapsed_time:.2f} seconds")
                    print(f"📈 Average step time: {elapsed_time / horizon:.3f} seconds")
                    print("=" * 80 + "\n")
                else:
                    print()

    predictions = np.array(predictions)

    if params.get('use_difference', False):
        predictions = de_difference(predictions, last_actual_value)

    if params.get('is_plot', False):
        plot_future_prediction(
            original_data,
            predictions,
            params,
            results_dir,
            is_in_future_prediction=is_in_future_prediction,
            date_time=date_time
        )

    return predictions

def prepare_data_for_prediction(data_source: str, params: dict, horizon: Optional[int], logger, verbose: str):
    """
    Fetches new data from the given data_source and applies all preprocessing.
    """
    if horizon is not None:
        params['horizon'] = horizon

    new_data, date_time,data_mode = fetch_data(data_source,slice_range=slice(-params['window_size'], None))
    original_data = new_data.copy()

    # if params['is_down_sample']:
    #     new_data = progressive_downsample(data=new_data,rate=params['down_sample_rate'])
    #
    # if params['imputation_method']:
    #     new_data = impute_missing_values(
    #         new_data,
    #         method=params['imputation_method'],
    #         n_neighbors=params['imputation_neighbors'],
    #         logger=logger
    #     )
    #
    # if params.get('use_difference', False):
    #     new_data = compute_difference(new_data, params=params, is_plot=False)
    #
    # if params['outlier_removal']:
    #     new_data = remove_outliers(
    #         new_data,
    #         method=params['outlier_method'],
    #         params=params
    #     )
    #
    scaler = None
    # if params['normalization_method'] and params.get('normalization_type', None) == 'all':
    #     scaler = params.get('scaler')
    #     if scaler:
    #         new_data = scaler.transform(new_data)
    #     else:
    #         new_data, _ = normalize_data(
    #             new_data,
    #             method=params['normalization_method'],
    #             logger=logger
    #         )

    if len(new_data.shape) == 1 or new_data.shape[1] == 1:
        new_data = new_data.reshape(-1, 1)

    return new_data, original_data, date_time, scaler, data_mode

def future_prediction_with_new_data(
        model_path: Optional[str] = None,
        params_path: Optional[str] = None,
        model: Optional[tf.keras.Model] = None,
        loaded_params: Optional[dict] = None,
        data_source: Optional[str] = None,
        horizon: Optional[int] = None,
        verbose: str = "all"
) -> tuple:
    """
    Perform future prediction using new data with adjustable verbosity.
    Returns (predictions, new_data, date_time).
    """
    if data_source is None:
        raise ValueError("You must provide a data_source for new data.")

    logger = logging.getLogger('TimeSeriesModel')
    if verbose != "off":
        logger.info("Performing future prediction with new data")

    if model is None or loaded_params is None:
        if model_path is None or params_path is None:
            raise ValueError(
                "Either provide a pre-loaded model & params, "
                "OR specify both model_path and params_path."
            )
        model, loaded_params = load_model_and_params(model_path, params_path)

    params = loaded_params
    # Force no-plot to ensure we don't generate plots in the periodic task
    params['is_plot'] = False
    params['is_in_training'] = False
    params['is_down_sample'] = params.get('is_down_sample',False)

    new_data, original_data, date_time, scaler, data_mode = prepare_data_for_prediction(
        data_source=data_source,
        params=params,
        horizon=horizon,
        logger=logger,
        verbose=verbose
    )

    predictions = future_prediction(
        model=model,
        data=new_data,
        original_data=original_data,
        params=params,
        scaler=scaler if params.get('normalization_type', None) == 'all' else None,
        results_dir=params.get('results_save_dir'),
        is_in_future_prediction=True,
        verbose=verbose,
        date_time=date_time
    )

    if data_mode=='offline':
        plot_future_prediction(original_data,
                               predictions,
                               params,
                               params.get('results_save_dir'),
                               is_in_future_prediction=True,
                               date_time=None)

    if verbose != "off":
        logger.info(f"Future Predictions (new data): {predictions}")

    return predictions, new_data, date_time

app = FastAPI()

@app.get("/predict")
def predict_endpoint():
    """
    When called, returns the latest predictions from the periodic task.
    """
    global latest_preds
    if latest_preds is None:
        return {"predictions": None}
    return {"predictions": latest_preds.tolist()}
# ---------------------------------------------------------------------
# New endpoints to get the latest data, date_time, predictions,
# and the time of the last prediction update
# ---------------------------------------------------------------------

@app.get("/get_data")
def get_data():
    """
    Returns the latest new_data and date_time.
    If not available, returns None.
    """
    global latest_new_data, latest_date_time

    if latest_new_data is None or latest_date_time is None:
        return {"new_data": None, "date_time": None}

    # Convert latest_new_data to a 1D list
    new_data_list = latest_new_data.ravel().tolist()

    # Convert latest_date_time (DatetimeIndex) to a list of strings
    date_time_list = latest_date_time.strftime('%Y-%m-%dT%H:%M:%S').tolist()

    return {
        "new_data": new_data_list,
        "date_time": date_time_list
    }

@app.get("/get_prediction")
def get_prediction():
    """
    Returns the latest predictions.
    If not available, returns None.
    """
    global latest_preds
    if latest_preds is None:
        return {"predictions": None}

    return {"predictions": latest_preds.ravel().tolist()}

@app.get("/get_prediction_time")
def get_prediction_time():
    """
    Returns the latest update time for the prediction.
    If not available, returns None.
    """
    global latest_pred_update_time
    if latest_pred_update_time is None:
        return {"latest_prediction_update_time": None}
    return {"latest_prediction_update_time": latest_pred_update_time}
# ---------------------------------------------------------------------
def run_periodic_prediction():
    """
    Periodically fetches new data and runs future_prediction_with_new_data every 15 seconds,
    updating global variables with the latest predictions and data.
    current est models:
        -xauusd:  run_20250107_150553
        -btcusdt: run_20250999_182305
    """
    global latest_preds, latest_new_data, latest_date_time, latest_pred_update_time

    base_dir = "../results/run_20250223_161425" # xauusd
    # base_dir = "../results/run_20250319_202304" # btcusdt
    model_path = base_dir
    params_path = base_dir


    while True:
        try:
            preds, new_data, dt = future_prediction_with_new_data(
                model_path=model_path,
                params_path=params_path,
                data_source="http://localhost:6000/data",
                horizon=90,
                verbose="short"
            )
            latest_preds = preds
            latest_new_data = new_data
            latest_date_time = dt
            latest_pred_update_time = time.ctime()  # or time.time(), whichever you prefer
        except Exception as e:
            print(f"Periodic prediction error: {e}")

        time.sleep(15)

if __name__ == "__main__":
    # Run the periodic prediction in a separate thread
    threading.Thread(target=run_periodic_prediction, daemon=True).start()
    while True:
        try:
            uvicorn.run(app, host="0.0.0.0", port=5500)
        except Exception as exc:
            print(f"Server crashed with error: {exc}")
            print("Re-starting server...")
            time.sleep(3)
