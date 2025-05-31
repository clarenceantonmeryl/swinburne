import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
                                     BatchNormalization, Bidirectional, Attention, GlobalAveragePooling1D)
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

# Merge Streams + Add Magnitudes
def merge_streams(worker_id, tolerance=0.01):
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

    merged = pd.merge_asof(df_acc_r, df_gyro_r, on='time', direction='nearest', tolerance=tolerance)
    merged = pd.merge_asof(merged, df_acc_l, on='time', direction='nearest', tolerance=tolerance)
    merged = pd.merge_asof(merged, df_gyro_l, on='time', direction='nearest', tolerance=tolerance)

    merged = merged.dropna(subset=['acc_r_x', 'gyro_r_x', 'acc_l_x', 'gyro_l_x'])

    print(f"Merged shape (worker {worker_id}): {merged.shape}")
    return merged

# Create Sliding Windows
def create_windows(data, window_size, step_size):
    X, y = [], []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        window = data.iloc[start:end]
        if window['class'].nunique() > 1:
            continue
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

# CNN-BiLSTM + Attention Model
input_shape = (window_size, 12)

# Use Functional API for Attention
inputs = Input(shape=input_shape)
x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Bidirectional(LSTM(100, return_sequences=True))(x)

# Add Attention layer
attention = Attention()([x, x])

# Option: Global Pooling to flatten the sequence
x = GlobalAveragePooling1D()(attention)

x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

# Train
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n Test Accuracy: {acc:.3f}")
