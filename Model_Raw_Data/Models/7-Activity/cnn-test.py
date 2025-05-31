import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
import os

# CONFIG
window_size = 30
step_size = 15
data_dir = '../../Data/7-Activity'
workers = [1, 2, 3, 4]
classes = 7  # Number of activities

# Load and Preprocess Function
def load_and_preprocess_df(path):
    df = pd.read_csv(path)
    df[['x', 'y', 'z']] = StandardScaler().fit_transform(df[['x', 'y', 'z']])
    return df

# Load All Datasets
datasets = {}
for w in workers:
    datasets[f'w{w}_right_acc'] = load_and_preprocess_df(os.path.join(data_dir, f'w{w}_right/acc_right_annotated.csv'))
    datasets[f'w{w}_right_gyro'] = load_and_preprocess_df(os.path.join(data_dir, f'w{w}_right/gyro_right_annotated.csv'))
    datasets[f'w{w}_left_acc'] = load_and_preprocess_df(os.path.join(data_dir, f'w{w}_left/acc_left_annotated.csv'))
    datasets[f'w{w}_left_gyro'] = load_and_preprocess_df(os.path.join(data_dir, f'w{w}_left/gyro_left_annotated.csv'))


# Merge Streams (Align by Time)
def merge_streams(worker_id, tolerance=0.02):
    # Prepare dataframes and rename columns
    df_acc_r = datasets[f'w{worker_id}_right_acc'].rename(
        columns={'x': 'acc_r_x', 'y': 'acc_r_y', 'z': 'acc_r_z'}).sort_values('time')

    df_gyro_r = datasets[f'w{worker_id}_right_gyro'].rename(
        columns={'x': 'gyro_r_x', 'y': 'gyro_r_y', 'z': 'gyro_r_z'}).sort_values('time')
    df_gyro_r = df_gyro_r.drop(columns=['class'])

    df_acc_l = datasets[f'w{worker_id}_left_acc'].rename(
        columns={'x': 'acc_l_x', 'y': 'acc_l_y', 'z': 'acc_l_z'}).sort_values('time')
    df_acc_l = df_acc_l.drop(columns=['class'])

    df_gyro_l = datasets[f'w{worker_id}_left_gyro'].rename(
        columns={'x': 'gyro_l_x', 'y': 'gyro_l_y', 'z': 'gyro_l_z'}).sort_values('time')
    df_gyro_l = df_gyro_l.drop(columns=['class'])

    # Step 1: merge acc_r and gyro_r by nearest timestamp
    merged = pd.merge_asof(df_acc_r, df_gyro_r, on='time', direction='nearest', tolerance=tolerance)

    # Step 2: merge in acc_l
    merged = pd.merge_asof(merged, df_acc_l, on='time', direction='nearest', tolerance=tolerance)

    # Step 3: merge in gyro_l
    merged = pd.merge_asof(merged, df_gyro_l, on='time', direction='nearest', tolerance=tolerance)

    # Drop any rows where core sensors are still missing (if no nearby match found)
    merged = merged.dropna(subset=['acc_r_x', 'gyro_r_x', 'acc_l_x', 'gyro_l_x'])
    print(merged.shape)

    return merged


# Create Sliding Windows
def create_windows(data, window_size, step_size):
    X, y = [], []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        window = data.iloc[start:end]
        if window['class'].nunique() > 1:
            continue  # Skip windows with mixed labels
        X.append(window.drop(columns=['time', 'class']).values)
        y.append(window['class'].iloc[0])
    return np.array(X), np.array(y)


# Process All Workers
X_all, y_all = [], []

for w in workers:
    merged_df = merge_streams(w)
    X, y = create_windows(merged_df, window_size, step_size)
    X_all.append(X)
    y_all.append(y)

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print(f"Final dataset shape: {X_all.shape}, Labels shape: {y_all.shape}")

# CNN-LSTM Model
input_shape = (window_size, 12)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=False),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

# Train
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {acc:.3f}")
