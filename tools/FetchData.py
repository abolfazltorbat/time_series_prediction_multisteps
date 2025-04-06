import numpy as np
import pandas as pd
import requests
from tools.PreProcessing import load_csv_data
import logging
from typing import Tuple, Optional


def fetch_data(data_source: str, slice_range: Optional[slice] = slice(200, None)) -> Tuple[np.ndarray, Optional[pd.DatetimeIndex], Optional[str]]:
    """
    Fetch data from either a file path or an HTTP address with optional range slicing.

    Args:
        data_source: URL or file path to fetch data from
        slice_range: Optional slice object to select data range (default: None for all data)

    Returns:
        Tuple of (close_values array, datetime index if available)

    Raises:
        ConnectionError: If HTTP request fails
    """
    if not any(data_source.startswith(prefix) for prefix in ('http://', 'https://', '127.0.0.1')):
        return load_csv_data(data_source, logging), None, 'offline'

    try:
        url = f'http://{data_source}' if data_source.startswith('127') else data_source
        response = requests.get(url).json()
        fields = ['Close', 'Gmt time', 'High', 'Low', 'Open', 'Symbol', 'Timeframe', 'Volume']
        data = {
            field: (response.get(field, [None])[slice_range] if slice_range
                    else response.get(field, [None]))
            for field in fields
        }

        close_values = np.array(data['Close'], dtype=float).reshape(-1, 1)

        if data['Gmt time']:
            try:
                date_time = pd.to_datetime(data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
            except ValueError:
                # Fallback for different datetime formats
                date_time = pd.to_datetime(data['Gmt time'])
        else:
            date_time = None

        return close_values, date_time, 'online'

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to fetch data from {data_source}: {str(e)}")