from __future__ import annotations

import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model_training.data_pipeline import (
    add_pca_features,
    add_target_columns,
    build_sequences,
    load_panel_dataframe,
    split_by_date,
)

SEED = 42
SEQUENCE_LENGTH = 12
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = tf.keras.layers.LSTM(128, dropout=0.3, recurrent_dropout=0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def main() -> None:
    set_seed()
    panel_df = load_panel_dataframe()
    panel_df = add_target_columns(panel_df, target_col="net_sales")

    feature_cols = [col for col in panel_df.columns if col not in {"company", "date", "target_value", "target_date"}]
    panel_df, artifacts = add_pca_features(panel_df, feature_cols, train_end="2022-01-01")
    pc_cols = artifacts.component_names

    sequences, targets, target_dates = build_sequences(
        panel_df,
        pc_cols,
        sequence_length=SEQUENCE_LENGTH,
    )
    train_mask, val_mask, test_mask = split_by_date(
        target_dates, train_end=TRAIN_END, val_end=VAL_END
    )
    X_train, y_train = sequences[train_mask], targets[train_mask]
    X_val, y_val = sequences[val_mask], targets[val_mask]
    X_test, y_test = sequences[test_mask], targets[test_mask]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ARTIFACT_DIR / "cnn_lstm_best.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    best_model = tf.keras.models.load_model(ARTIFACT_DIR / "cnn_lstm_best.keras")
    y_pred = best_model.predict(X_test).flatten()
    metrics = evaluate_predictions(y_test, y_pred)

    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(ARTIFACT_DIR / "training_history.csv", index=False)
    joblib.dump(artifacts.scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(artifacts.pca, ARTIFACT_DIR / "pca.joblib")
    best_model.save(ARTIFACT_DIR / "cnn_lstm_model.keras")


if __name__ == "__main__":
    main()

