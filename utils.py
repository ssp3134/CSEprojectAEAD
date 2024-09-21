import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random 
from sklearn.preprocessing import StandardScaler

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)

class SyntheticTimeSeries(Dataset):
    def _make_chunks(self, data, window_size):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        scaled_data = torch.from_numpy(scaled_data)
        return scaled_data.unfold(0, window_size, window_size)
    
    def _generate_data(
            self,
            start_time, 
            periods, 
            freq='min', 
            trend_amplitude=0.5, 
            seasonality_amplitude=5, 
            noise_amplitude=1, 
            anomaly_factor=10, 
            num_anomalies=5
        ):
        """
        Generates a minute-level time series with anomalies.

        Parameters:
        - start_time (str): Start time of the time series.
        - periods (int): Number of periods (data points) to generate.
        - freq (str): Frequency string (e.g., '1T' for 1 minute).
        - trend_amplitude (float): Amplitude of the trend component.
        - seasonality_amplitude (float): Amplitude of the seasonal component.
        - noise_amplitude (float): Amplitude of the noise component.
        - anomaly_factor (float): Factor by which anomalies deviate from normal data.
        - num_anomalies (int): Number of anomalies to introduce.

        Returns:
        - time_series_df (DataFrame): DataFrame containing the time series.
        - anomaly_indices (list): Indices where anomalies are located.
        """
        # Create a time range starting from the provided start time
        time_range = pd.date_range(start=start_time, periods=periods, freq=freq)

        # Generate a random trend component (random walk)
        trend_component = np.cumsum(np.random.randn(periods) * trend_amplitude)

        # Generate a seasonal component (sinusoidal pattern)
        seasonal_period = 60 * 6  # 6-hour repeating seasonality
        seasonality_component = seasonality_amplitude * np.sin(
            np.linspace(0, 2 * np.pi * periods / seasonal_period, periods)
        )

        # Generate random noise component
        noise_component = noise_amplitude * np.random.randn(periods)

        # Combine trend, seasonality, and noise to form the base time series
        time_series_values = trend_component + seasonality_component + noise_component

        # Introduce anomalies at random points in the time series
        anomaly_indices = random.sample(range(periods), num_anomalies)

        for idx in anomaly_indices:
            # Randomly choose whether to make the anomaly a spike (positive) or a dip (negative)
            if random.choice([True, False]):
                time_series_values[idx] += anomaly_factor * noise_amplitude  # Positive anomaly (spike)
            else:
                time_series_values[idx] -= anomaly_factor * noise_amplitude  # Negative anomaly (dip)

        # Shift all values to be positive
        lowest_value = np.min(time_series_values)
        if lowest_value < 0:
            time_series_values += abs(lowest_value) + 1  # Shift all values to make them positive

        # Create a DataFrame with 'Timestamp' and 'Value'
        time_series_df = pd.DataFrame({'timestamp': time_range, 'value': time_series_values})

        return time_series_df
        
        
    def __init__(
        self,
        periods, 
        freq='min', 
        trend_amplitude=0.5, 
        seasonality_amplitude=5, 
        noise_amplitude=1, 
        anomaly_factor=10, 
        num_anomalies=5,
        window_size=60,
    ):
        super().__init__()
        assert periods >= window_size, "window size must be greater than number of periods"
        self.periods = periods
        start_time = "2023-09-21 09:00"
        self.df = self._generate_data(
            start_time,
            self.periods,
            freq, 
            trend_amplitude, 
            seasonality_amplitude, 
            noise_amplitude, 
            anomaly_factor, 
            num_anomalies
        )
        self.window_size = window_size
        chunks = self._make_chunks(self.df.value.values, self.window_size)
        self.chunks = chunks.to(dtype=torch.float16)
    
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return self.chunks[idx]
    
def get_time_series_data(
        periods, 
        batch_size=32,
        freq='min', 
        trend_amplitude=0.5, 
        seasonality_amplitude=5, 
        noise_amplitude=1, 
        anomaly_factor=10, 
        num_anomalies=5,
        window_size=60,
        val_size=None
    ):
    periods = int(periods)
    time_series = SyntheticTimeSeries(
        periods, 
        freq=freq, 
        trend_amplitude=trend_amplitude, 
        seasonality_amplitude=seasonality_amplitude, 
        noise_amplitude=noise_amplitude, 
        anomaly_factor=anomaly_factor, 
        num_anomalies=num_anomalies,
        window_size=window_size,
    )
    if val_size is None:
        return DataLoader(time_series, batch_size=batch_size, shuffle=False)
    val_split_size = int(val_size * len(time_series.chunks))
    train_data = time_series[: -val_split_size]
    test_data = time_series[-val_split_size:]
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=False),
        DataLoader(test_data, batch_size=batch_size, shuffle=False)
    )

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")

def generate_random_time_series(num=100, window_size=60, device="cpu"):
    x = torch.randn(num, window_size)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x = torch.from_numpy(x_scaled)
    return x.to(device, dtype=torch.float16)