import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
                                     BatchNormalization, Attention, GlobalAveragePooling1D, Concatenate, Bidirectional)
import os

# CONFIG
window_size = 30
step_size = 15
data_dir = '../../Data/3-Activity'
workers = [1, 2, 3, 4]
classes = 3  # Number of activities

# Load & Preprocess Function
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

# Merge Streams (Separate Left and Right Streams)
def merge_streams(worker_id, tolerance=0.02):
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

# Create Sliding Windows (Separate Left and Right Streams)
def create_windows_dual(data, window_size, step_size):
    X_right, X_left, y = [], [], []
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        window = data.iloc[start:end]
        if window['class'].nunique() > 1:
            continue
        right_feats = window[['acc_r_x', 'acc_r_y', 'acc_r_z',
                              'gyro_r_x', 'gyro_r_y', 'gyro_r_z',
                              ]].values

        left_feats = window[['acc_l_x', 'acc_l_y', 'acc_l_z',
                             'gyro_l_x', 'gyro_l_y', 'gyro_l_z',
                             ]].values

        X_right.append(right_feats)
        X_left.append(left_feats)
        y.append(window['class'].iloc[0])
    return np.array(X_right), np.array(X_left), np.array(y)

# Process All Workers
X_right_all, X_left_all, y_all = [], [], []

for w in workers:
    merged_df = merge_streams(w)
    X_r, X_l, y = create_windows_dual(merged_df, window_size, step_size)
    X_right_all.append(X_r)
    X_left_all.append(X_l)
    y_all.append(y)

X_right_all = np.concatenate(X_right_all, axis=0)
X_left_all = np.concatenate(X_left_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

print(f"Right shape: {X_right_all.shape}, Left shape: {X_left_all.shape}, Labels shape: {y_all.shape}")

# Dual-Stream CNN-LSTM + Cross-Attention Model
right_input = Input(shape=(window_size, 6), name='right_input')
left_input = Input(shape=(window_size, 6), name='left_input')

# Process Right Hand Stream
x_r = Conv1D(filters=64, kernel_size=3, activation='relu')(right_input)
x_r = BatchNormalization()(x_r)
x_r = MaxPooling1D(pool_size=2)(x_r)
x_r = Conv1D(filters=128, kernel_size=3, activation='relu')(x_r)
x_r = BatchNormalization()(x_r)
x_r = MaxPooling1D(pool_size=2)(x_r)
x_r = Bidirectional(LSTM(100, return_sequences=True))(x_r)

# Process Left Hand Stream
x_l = Conv1D(filters=64, kernel_size=3, activation='relu')(left_input)
x_l = BatchNormalization()(x_l)
x_l = MaxPooling1D(pool_size=2)(x_l)
x_l = Conv1D(filters=128, kernel_size=3, activation='relu')(x_l)
x_l = BatchNormalization()(x_l)
x_l = MaxPooling1D(pool_size=2)(x_l)
x_l = Bidirectional(LSTM(100, return_sequences=True))(x_l)

# Cross-Attention Layers
# Right attends to Left
att_r = Attention()([x_r, x_l])
# Left attends to Right
att_l = Attention()([x_l, x_r])

# Option: Pooling and Combine
pooled_r = GlobalAveragePooling1D()(att_r)
pooled_l = GlobalAveragePooling1D()(att_l)

# Combine both streams
combined = Concatenate()([pooled_r, pooled_l])
combined = Dropout(0.5)(combined)
combined = Dense(64, activation='relu')(combined)
output = Dense(classes, activation='softmax')(combined)

# Build Model
model = Model(inputs=[right_input, left_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train-Test Split
X_r_train, X_r_test, X_l_train, X_l_test, y_train, y_test = train_test_split(
    X_right_all, X_left_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

# Train
history = model.fit([X_r_train, X_l_train], y_train,
                    epochs=100, batch_size=64,
                    validation_data=([X_r_test, X_l_test], y_test))

# Evaluate
loss, acc = model.evaluate([X_r_test, X_l_test], y_test)
print(f"\n Test Accuracy: {acc:.3f}")
