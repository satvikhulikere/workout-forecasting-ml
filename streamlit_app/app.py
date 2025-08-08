import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

# Constants and config
DATA_PATH = "../workout-forecasting-ml/data/workout_data.csv"
MODEL_DIR = "../workout-forecasting-ml"
FEATURE_COLS = ['Reps', 'Sets', 'SleepHours', 'ProteinIntake', 'Calories', 'SorenessLevel', 'Bodyweight']
TARGET_COL = 'WeightLifted'
SEQUENCE_LENGTH = 5

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Exercise', 'Date']).reset_index(drop=True)
    return df

df = load_data()

st.title("Workout Weight Lifted Forecasting")

exercise_list = df["Exercise"].unique().tolist()
selected_exercise = st.sidebar.selectbox("Select Exercise", exercise_list)

# Filter data for exercise and drop missing rows
ex_df = df[df['Exercise'] == selected_exercise].reset_index(drop=True)
ex_df_clean = ex_df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

model_path = os.path.join(MODEL_DIR, f"{selected_exercise}.h5")
if not os.path.exists(model_path):
    st.error(f"Model for {selected_exercise} not found at {model_path}. Please train it first.")
    st.stop()

model = load_model(model_path)

# Scale features and target using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(ex_df_clean[FEATURE_COLS + [TARGET_COL]])
scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLS + [TARGET_COL])

features_scaled = scaled_df[FEATURE_COLS].values
target_scaled = scaled_df[TARGET_COL].values

def create_sequences(features, target, seq_length):
    x, y = [], []
    for i in range(len(target) - seq_length):
        x.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(x), np.array(y)

x, y = create_sequences(features_scaled, target_scaled, SEQUENCE_LENGTH)

# Show input and model shapes for debugging
st.write(f"Input data shape (x): {x.shape}")
st.write(f"Model input shape: {model.input_shape}")

# Check if shapes match, else stop
if x.shape[2] != model.input_shape[2]:
    st.error(f"Feature count mismatch! Model expects {model.input_shape[2]} features but input has {x.shape[2]}.")
    st.stop()

# Predict on historical data
preds_scaled = model.predict(x)
dummy = np.zeros((len(preds_scaled), len(FEATURE_COLS) + 1))
dummy[:, -1] = preds_scaled.flatten()
predicted_weights = scaler.inverse_transform(dummy)[:, -1]

dummy[:, -1] = y
actual_weights = scaler.inverse_transform(dummy)[:, -1]

# Plot Actual vs Predicted
st.subheader(f"{selected_exercise} — Actual vs Predicted Weight Lifted")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ex_df_clean['Date'][SEQUENCE_LENGTH:], actual_weights, label='Actual', marker='o')
ax.plot(ex_df_clean['Date'][SEQUENCE_LENGTH:], predicted_weights, label='Predicted', linestyle='--', marker='x')
ax.set_xlabel("Date")
ax.set_ylabel("Weight Lifted (lbs)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Forecast section
st.subheader("Forecast Weight Lifted for a Future Date")

future_date = st.date_input("Select future date", min_value=ex_df_clean['Date'].max().date())

if future_date <= ex_df_clean['Date'].max().date():
    st.warning("Please select a future date beyond the latest date in the dataset.")
else:
    st.markdown("### Enter feature values for the forecast date:")

    # User inputs for the 7 features
    input_features = {}
    for feature in FEATURE_COLS:
        default_val = float(ex_df_clean[feature].median())
        input_features[feature] = st.number_input(f"{feature}", value=default_val, format="%.2f")

    # Prepare sequence for prediction:
    # Take last SEQUENCE_LENGTH-1 sequences from scaled features
    last_seq = features_scaled[-(SEQUENCE_LENGTH - 1):]

    # Scale the user input features using the existing scaler
    input_df = pd.DataFrame([input_features])
    # Add dummy target column (needed for scaler)
    input_scaled_full = scaler.transform(
        pd.concat([input_df, pd.DataFrame({TARGET_COL: [0]})], axis=1)
    )
    input_scaled_features = input_scaled_full[:, :-1]  # exclude target col

    # Construct the full input sequence for prediction
    new_seq = np.vstack([last_seq, input_scaled_features])
    new_seq = new_seq.reshape((1, SEQUENCE_LENGTH, len(FEATURE_COLS)))

    # Predict scaled
    pred_scaled = model.predict(new_seq)

    # Inverse scale prediction
    dummy_pred = np.zeros((1, len(FEATURE_COLS) + 1))
    dummy_pred[:, -1] = pred_scaled.flatten()
    pred_weight = scaler.inverse_transform(dummy_pred)[:, -1][0]

    st.markdown(f"### Forecasted Weight Lifted on {future_date}: **{pred_weight:.2f} lbs**")
