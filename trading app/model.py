"""
model.py
--------
Dual-head LSTM model:
  - Classification head: predicts price direction (UP/DOWN)
  - Regression head: predicts next candle close price

Architecture:
  Input → 2× LSTM(128) with dropout → Dense(64) →
    ├─ Direction head: Dense(1, sigmoid)  [binary crossentropy]
    └─ Price head:    Dense(1, linear)    [MSE]

Training uses early stopping and learning-rate scheduling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import joblib
import os

from data_pipeline import FEATURE_COLUMNS


class PricePredictor:
    def __init__(self, lookback: int = 60, model_dir: str = "saved_model"):
        """
        Args:
            lookback:  Number of past candles fed into the LSTM.
            model_dir: Directory to save/load model weights and scaler.
        """
        self.lookback  = lookback
        self.model_dir = model_dir
        self.scaler    = RobustScaler()   # Robust to outliers (spikes)
        self.model     = None
        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_sequences(self, df: pd.DataFrame, fit_scaler: bool = True):
        """
        Scale features and create (lookback × n_features) sequences.

        Returns:
            X: shape (samples, lookback, n_features)
            y_dir: direction labels  (samples,)
            y_price: next-close values (samples,)
        """
        feature_data = df[FEATURE_COLUMNS].values
        y_dir   = df["target_direction"].values
        y_price = df["target_next_close"].values

        if fit_scaler:
            feature_data = self.scaler.fit_transform(feature_data)
            joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        else:
            feature_data = self.scaler.transform(feature_data)

        X, yd, yp = [], [], []
        for i in range(self.lookback, len(feature_data)):
            X.append(feature_data[i - self.lookback : i])
            yd.append(y_dir[i])
            yp.append(y_price[i])

        return np.array(X), np.array(yd), np.array(yp)

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------

    def build_model(self, n_features: int) -> Model:
        """
        Dual-head LSTM.

        Input shape: (lookback, n_features)
        """
        inp = layers.Input(shape=(self.lookback, n_features), name="candle_sequence")

        # Shared LSTM backbone
        x = layers.LSTM(128, return_sequences=True, name="lstm_1")(inp)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128, return_sequences=False, name="lstm_2")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu", name="shared_dense")(x)
        x = layers.BatchNormalization()(x)

        # Direction head (classification)
        dir_out = layers.Dense(32, activation="relu")(x)
        dir_out = layers.Dense(1, activation="sigmoid", name="direction")(dir_out)

        # Price head (regression)
        price_out = layers.Dense(32, activation="relu")(x)
        price_out = layers.Dense(1, activation="linear", name="next_price")(price_out)

        model = Model(inputs=inp, outputs=[dir_out, price_out])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                "direction":  "binary_crossentropy",
                "next_price": "mse"
            },
            loss_weights={
                "direction":  1.0,   # prioritise direction accuracy
                "next_price": 0.3
            },
            metrics={
                "direction":  ["accuracy"],
                "next_price": ["mae"]
            }
        )
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train with TimeSeriesSplit cross-validation (no data leakage).
        Final model is fit on all available data.

        Returns:
            Dictionary with validation accuracy and loss history.
        """
        X, y_dir, y_price = self.prepare_sequences(df, fit_scaler=True)
        n_features = X.shape[2]

        print(f"Training on {X.shape[0]} sequences, {n_features} features, lookback={self.lookback}")

        # ---- Cross-validation (walk-forward) ----
        tscv = TimeSeriesSplit(n_splits=5)
        val_accs = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n--- Fold {fold + 1}/5 ---")
            model = self.build_model(n_features)
            cb = [
                callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_direction_accuracy"),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)
            ]
            history = model.fit(
                X[train_idx], {"direction": y_dir[train_idx], "next_price": y_price[train_idx]},
                validation_data=(X[val_idx], {"direction": y_dir[val_idx], "next_price": y_price[val_idx]}),
                epochs=epochs, batch_size=batch_size,
                callbacks=cb, verbose=0
            )
            best_val_acc = max(history.history["val_direction_accuracy"])
            val_accs.append(best_val_acc)
            print(f"Best val direction accuracy: {best_val_acc:.4f}")

        print(f"\nMean CV direction accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")

        # ---- Final model on all data ----
        print("\nTraining final model on full dataset...")
        self.model = self.build_model(n_features)
        cb_final = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="direction_accuracy"),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, "model.keras"),
                save_best_only=True, monitor="direction_accuracy"
            )
        ]
        final_history = self.model.fit(
            X, {"direction": y_dir, "next_price": y_price},
            epochs=epochs, batch_size=batch_size,
            callbacks=cb_final, verbose=1
        )
        return {
            "cv_accuracy_mean": float(np.mean(val_accs)),
            "cv_accuracy_std":  float(np.std(val_accs)),
            "history": final_history.history
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Run inference on the most recent `lookback` candles.

        Returns:
            {
              "direction":   "UP" | "DOWN",
              "confidence":  float 0-1,
              "next_price":  float,
              "signal":      "BUY" | "SELL" | "HOLD"  (based on confidence)
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Run train() or load_model() first.")

        recent = df.tail(self.lookback + 50)   # +50 for indicator warmup
        recent = recent[FEATURE_COLUMNS].values
        recent_scaled = self.scaler.transform(recent)
        seq = recent_scaled[-self.lookback:]   # (lookback, n_features)
        seq = seq.reshape(1, self.lookback, recent_scaled.shape[1])

        dir_prob, price_pred = self.model.predict(seq, verbose=0)
        confidence   = float(dir_prob[0][0])
        next_price   = float(price_pred[0][0])
        direction    = "UP" if confidence > 0.5 else "DOWN"

        return {
            "direction":  direction,
            "confidence": confidence if direction == "UP" else 1 - confidence,
            "next_price": next_price,
            "raw_prob":   confidence
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        self.model.save(os.path.join(self.model_dir, "model.keras"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        print(f"Model saved to {self.model_dir}/")

    def load(self):
        model_path  = os.path.join(self.model_dir, "model.keras")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No saved model at {model_path}. Train first.")
        self.model  = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded.")
