"""Generates realistic synthetic hourly pollution data."""
import numpy as np
import pandas as pd

STATIONS = [  # name, city, lat, lon, pollution level multiplier
    ("Zoo Park", "Hyderabad", 17.3507, 78.4510, 1.0),
    ("Sanathnagar", "Hyderabad", 17.4560, 78.4440, 1.2),
    ("Anand Vihar", "Delhi", 28.6469, 77.3160, 2.6),
    ("RK Puram", "Delhi", 28.5633, 77.1869, 2.0),
    ("Bandra", "Mumbai", 19.0596, 72.8295, 1.1),
    ("Peenya", "Bengaluru", 13.0285, 77.5197, 0.9),
    ("Alipore", "Kolkata", 22.5381, 88.3311, 1.6),
    ("T Nagar", "Chennai", 13.0418, 80.2341, 0.8),
]
BASE = {"PM2.5": 55, "PM10": 110, "NO2": 35, "SO2": 12, "CO": 1.1, "O3": 40}


def generate(days=730, seed=42):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days * 24, freq="h")
    doy, hr = ts.dayofyear.values, ts.hour.values
    season = 1 + 0.55 * np.cos(2 * np.pi * (doy - 15) / 365)
    rush = 1 + 0.35 * (np.exp(-((hr - 9) ** 2) / 8) + np.exp(-((hr - 20) ** 2) / 8))
    frames = []
    for name, city, lat, lon, mult in STATIONS:
        for pol, base in BASE.items():
            s = season if pol != "O3" else 2 - season
            d = rush if pol != "O3" else 1 + 0.5 * np.exp(-((hr - 14) ** 2) / 10)
            noise = rng.lognormal(0, 0.25, len(ts))
            drift = 1 + 0.00015 * np.arange(len(ts)) / 24 * (mult - 0.8)
            val = base * mult * s * d * noise * drift
            frames.append(pd.DataFrame({"timestamp": ts, "station_name": name, "city": city,
                                        "lat": lat, "lon": lon, "pollutant": pol,
                                        "value": val.round(2)}))
    return pd.concat(frames, ignore_index=True)