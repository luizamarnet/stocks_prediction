"""MLP classifier and regressor with a manual learning-rate schedule."""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, root_mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def _recreate_mlp_with_lr(
    old_mlp: MLPClassifier,
    new_lr: float,
    X_fit_scaled: np.ndarray,
    y_fit_array: np.ndarray,
    hidden: tuple[int, ...],
    alpha: float,
    seed: int,
) -> MLPClassifier:
    """Create an MLP with a new learning rate and copy the weights of the previous model.

    Note: currently unused (the call sites are commented out in the training loop).
    """
    new_mlp = MLPClassifier(hidden_layer_sizes=hidden, solver="sgd",
                            learning_rate_init=new_lr, learning_rate="constant",
                            alpha=alpha, momentum=0.9, nesterovs_momentum=True,
                            random_state=seed)
    # A single call initialises coefs_ and the optimiser with the new learning rate.
    # The update made by this single sample is discarded right after.
    new_mlp.partial_fit(X_fit_scaled[:1], y_fit_array[:1], classes=[0, 1])
    new_mlp.coefs_ = [c.copy() for c in old_mlp.coefs_]
    new_mlp.intercepts_ = [b.copy() for b in old_mlp.intercepts_]
    return new_mlp


class ScaledClassifier:
    """Wrap a scaler and an MLP classifier, exposing predict / predict_proba."""

    def __init__(self, scaler: StandardScaler, mlp: MLPClassifier) -> None:
        self.scaler = scaler
        self.mlp = mlp

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities for the input features."""
        return self.mlp.predict_proba(self.scaler.transform(X))

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return predicted class labels for the input features."""
        return self.mlp.predict(self.scaler.transform(X))


class ScaledRegressor:
    """Wrap the X scaler, the y scaler and the MLP. ``predict`` returns y in its original scale."""

    def __init__(self, scaler_x: StandardScaler, scaler_y: StandardScaler, mlp: MLPRegressor) -> None:
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.mlp = mlp

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return regression predictions in the original target scale."""
        X_scaled = self.scaler_x.transform(X)
        y_scaled = self.mlp.predict(X_scaled).reshape(-1, 1)
        return self.scaler_y.inverse_transform(y_scaled).ravel()


def classifier_train(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    n_epochs: int = 2000,
    val_frac: float = 0.1,
    seed: int = 42,
    hidden: tuple[int, ...] = (64, 32),
    alpha: float = 1e-4,
    lr: float = 1e-3,
    patience: int = 50,
    decay_factor: float = 0.9,
    lr_min: float = 1e-5,
) -> tuple[ScaledClassifier, int, float, list[float]]:
    """Train an MLP classifier with chronological validation and a decaying learning rate.

    The last ``val_frac`` of the training rows is used as validation. The model
    with the lowest validation log loss is kept.

    Returns:
        Tuple of (trained model, best epoch, best validation log loss, validation loss history).

    Raises:
        ValueError: If the validation log loss becomes non-finite.
    """
    n_val = int(len(X_tr) * val_frac)
    X_fit, y_fit = X_tr.iloc[:-n_val], y_tr.iloc[:-n_val]
    X_val, y_val = X_tr.iloc[-n_val:], y_tr.iloc[-n_val:]
    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)
    X_val_scaled = scaler.transform(X_val)
    y_fit_array = y_fit.to_numpy()
    y_val_array = y_val.to_numpy()
    mlp = MLPClassifier(hidden_layer_sizes=hidden, solver="sgd",
                        learning_rate_init=lr, learning_rate="constant",
                        momentum=0.9, nesterovs_momentum=True,
                        random_state=seed)
    lr_current = lr
    best_loss, best_epoch, best_model = np.inf, 0, None
    epochs_without_improvement = 0
    history: list[float] = []
    lr_history: list[float] = []
    lr_reductions: list[tuple[int, float]] = []
    for epoch in range(1, n_epochs + 1):
        mlp.partial_fit(X_fit_scaled, y_fit_array, classes=[0, 1])   # one epoch
        loss = log_loss(y_val_array, mlp.predict_proba(X_val_scaled), labels=[0, 1])
        if not np.isfinite(loss):
            raise ValueError(f"Log loss is not finite at epoch {epoch}. Check features and target.")
        history.append(loss)
        lr_history.append(lr_current)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_model = deepcopy(mlp)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

            # Exponential decay from lr to lr_min over n_epochs
            rate = (lr_min / lr) ** (1 / n_epochs)
            lr_current = lr * (rate ** epoch)
            #lr_current = np.maximum(lr_current * decay_factor, lr_min)
            mlp._optimizer.learning_rate = lr_current
        

     
    majority_baseline = y_fit.value_counts(normalize=True).max()
    print(f"baseline acc (majority class): {majority_baseline:.4f} | "
          f"best epoch: {best_epoch} | log loss: {best_loss:.4f} | "
          f"final lr: {lr_current:.2e}")
    print("learning rate reductions (epoch, new lr):", lr_reductions)
    return ScaledClassifier(scaler, best_model), best_epoch, best_loss, history


def regressor_train(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    n_epochs: int = 2000,
    val_frac: float = 0.1,
    seed: int = 42,
    hidden: tuple[int, ...] = (64, 32, 8),
    lr: float = 1e-3,
    patience: int = 50,
    decay_factor: float = 0.9,
    lr_min: float = 1e-5,
) -> tuple[ScaledRegressor, int, float, list[float]]:
    """Train an MLP regressor with chronological validation and a decaying learning rate.

    Returns:
        Tuple of (trained model, best epoch, best validation RMSE, validation RMSE history).

    Raises:
        ValueError: If the validation RMSE becomes non-finite.
    """
    n_val = int(len(X_tr) * val_frac)
    X_fit, y_fit = X_tr.iloc[:-n_val], y_tr.iloc[:-n_val]
    X_val, y_val = X_tr.iloc[-n_val:], y_tr.iloc[-n_val:]
    scaler_x = StandardScaler().fit(X_fit)
    scaler_y = StandardScaler().fit(y_fit.to_numpy().reshape(-1, 1))
    X_fit_scaled = scaler_x.transform(X_fit)
    X_val_scaled = scaler_x.transform(X_val)
    y_fit_scaled = scaler_y.transform(y_fit.to_numpy().reshape(-1, 1)).ravel()
    # sklearn uses a constant learning rate; the decay is applied manually below
    mlp = MLPRegressor(hidden_layer_sizes=hidden, solver="sgd",
                       learning_rate_init=lr, learning_rate="constant",
                       momentum=0.9, nesterovs_momentum=True,
                       random_state=seed)
    lr_current = lr
    best_loss, best_epoch, best_model = np.inf, 0, None
    epochs_without_improvement = 0
    history: list[float] = []
    lr_history: list[float] = []
    for epoch in range(1, n_epochs + 1):
        mlp.partial_fit(X_fit_scaled, y_fit_scaled)
        y_pred_val = scaler_y.inverse_transform(mlp.predict(X_val_scaled).reshape(-1, 1)).ravel()
        loss = root_mean_squared_error(y_val, y_pred_val)
        if not np.isfinite(loss):
            raise ValueError(f"Validation RMSE is not finite at epoch {epoch}.")
        history.append(loss)
        lr_history.append(lr_current)
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_model = deepcopy(mlp)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

            # Exponential decay from lr to lr_min over n_epochs
            rate = (lr_min / lr) ** (1 / n_epochs)
            lr_current = lr * (rate ** epoch)
            #lr_current = np.maximum(lr_current * decay_factor, lr_min)
            mlp._optimizer.learning_rate = lr_current
       

    baseline = root_mean_squared_error(y_val, np.full(len(y_val), y_fit.mean()))
    print(f"baseline: {baseline:.6f} | best epoch: {best_epoch} "
          f"RMSE: {best_loss:.6f} | final lr: {lr_current:.2e}")
    model = ScaledRegressor(scaler_x, scaler_y, best_model)
    return model, best_epoch, best_loss, history
