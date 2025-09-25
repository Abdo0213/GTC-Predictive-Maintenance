import pandas as pd
import os

# Load raw test file (space-separated)
df = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)

# Assign column names (2 IDs + 3 op settings + 21 sensors = 26 total)
cols = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
df.columns = cols

# ✅ Keep only the required features for your model
REQUIRED_FEATURES = [
    "op_setting_1", "op_setting_2",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21"
]

# Create output folder
os.makedirs("engines_csv", exist_ok=True)

# Split by engine_id
for engine_id, df_engine in df.groupby("engine_id"):
    df_engine_out = df_engine #df_engine[REQUIRED_FEATURES]
    filename = f"engines_csv/engine{engine_id}.csv"
    df_engine_out.to_csv(filename, index=False)
    print(f"✅ Saved {filename} with {len(df_engine_out)} rows")
