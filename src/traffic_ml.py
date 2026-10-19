"""PyTorch supervised regression of trip duration using observed trip records.

Demo observations are synthetic and NOT measured traffic. This learns a simple
travel-time proxy; there is no claim of real congestion forecasting.
"""
import numpy as np


def train_duration_model(records, epochs=140, seed=42):
    """records DataFrame columns: distance_km, departure_hour, observed_minutes.

    Returns model and training MAE, deliberately not a holdout claim.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    cols = ['distance_km', 'departure_hour', 'observed_minutes']
    if not set(cols).issubset(records.columns):
        raise ValueError(f'Training observations must contain {cols}')
    data = records[cols].to_numpy(dtype=np.float32)
    if len(data) < 20 or not np.isfinite(data).all() or (data[:, 0] <= 0).any() or (data[:, 2] <= 0).any() or ((data[:, 1] < 0) | (data[:, 1] > 23)).any():
        raise ValueError('Require >=20 valid records: distance>0, duration>0, hour 0..23')
    # Travel time baseline plus learned congestion multiplier; distance scaling in km.
    features = make_features(data[:, 0], data[:, 1])
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(data[:, 2:3] / data[:, 0:1], dtype=torch.float32)
    model = nn.Sequential(nn.Linear(4, 24), nn.ReLU(), nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, 1), nn.Softplus())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = model(x).numpy().reshape(-1) * data[:, 0]
    return model, float(np.mean(np.abs(predicted - data[:, 2])))


def make_features(distance_km, hour):
    d = np.asarray(distance_km, dtype=np.float32)
    h = np.asarray(hour, dtype=np.float32)
    return np.column_stack([d / 15.0, np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), np.ones_like(d)]).astype(np.float32)


def predict_duration_matrix(model, distances_km, hour):
    import torch
    n = len(distances_km)
    matrix = np.asarray(distances_km, dtype=float)
    ix = np.array([(i,j) for i in range(n) for j in range(n) if i != j])
    feature = make_features(matrix[ix[:,0],ix[:,1]], np.full(len(ix), hour))
    with torch.no_grad():
        per_km = model(torch.tensor(feature)).numpy().reshape(-1)
    result = np.zeros((n,n), dtype=float)
    result[ix[:,0],ix[:,1]] = np.maximum(0, per_km) * matrix[ix[:,0],ix[:,1]]
    return result


def simulated_observations(n=600, seed=42):
    """A deterministic *synthetic* demonstration dataset, not road measurements."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    km = rng.uniform(0.3, 12, n)
    hour = rng.integers(0, 24, n)
    rush = np.exp(-((hour-8)/2)**2) + np.exp(-((hour-17.5)/2.5)**2)
    minutes = km * (2.1 + 1.1*rush) + rng.normal(0, 0.45, n)
    return pd.DataFrame({'distance_km':km, 'departure_hour':hour, 'observed_minutes':np.maximum(minutes,0.1)})


def evaluate_duration_model(records, epochs=140, seed=42):
    """Reproducible 70/15/15 random split, with a held-out test and speed baseline.

    Valid for IID synthetic samples, not a substitute for time-based validation on
    real chronological trip records.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import torch
    cols = ['distance_km', 'departure_hour', 'observed_minutes']
    if not set(cols).issubset(records.columns) or len(records) < 30:
        raise ValueError('Expected >=30 observations with distance_km, departure_hour, observed_minutes')
    indices = np.arange(len(records))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed)
    train = records.iloc[train_idx]
    validation = records.iloc[val_idx]
    test = records.iloc[test_idx]
    model, train_mae = train_duration_model(train, epochs=epochs, seed=seed)
    def predict(frame):
        features = make_features(frame.distance_km.to_numpy(), frame.departure_hour.to_numpy())
        with torch.no_grad():
            rate = model(torch.tensor(features, dtype=torch.float32)).numpy().flatten()
        return rate * frame.distance_km.to_numpy()
    # Honest baseline: constant minutes/km, fitted only on training data.
    baseline_rate = float(train.observed_minutes.sum() / train.distance_km.sum())
    def metrics(frame):
        y = frame.observed_minutes.to_numpy()
        return {'model_mae':float(mean_absolute_error(y, predict(frame))),
                'baseline_mae':float(mean_absolute_error(y, baseline_rate * frame.distance_km.to_numpy()))}
    return model, {'train':metrics(train), 'validation':metrics(validation), 'test':metrics(test),
                   'sizes':{'train':len(train),'validation':len(validation),'test':len(test)},
                   'split_note':'Random split for IID synthetic demo; chronological holdout needed for time-ordered real records.'}
