# src/models/mlp_keras.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, mixed_precision
from models.base_model import BaseModel

mixed_precision.set_global_policy("mixed_float16")

def build_keras_mlp(hidden_layer_sizes=(64, 32, 16), activation='relu',
                    solver='adam', alpha=0.001, learning_rate_init=0.001, input_shape=None):
    """
    Build a Keras MLP model for regression.
    
    Parameters:
      hidden_layer_sizes: tuple, number of units per hidden layer.
      activation: string, activation function to use.
      solver: string, optimizer name.
      alpha: float, L2 regularization strength.
      learning_rate_init: float, learning rate for the optimizer.
      input_shape: tuple specifying the shape of the input (required for the first layer).
    
    Returns:
      A compiled tf.keras model.
    """
    if input_shape is None:
        raise ValueError("input_shape must be provided for the model.")
    
    # Use an explicit Input layer for defining input shape
    model = models.Sequential([
        tf.keras.Input(shape=input_shape)
    ])
    
    # Add hidden layers
    for units in hidden_layer_sizes:
        model.add(layers.Dense(units,
                               activation=activation,
                               kernel_regularizer=regularizers.l2(alpha)))
    
    # Output layer
    # Output layer (force float32 to prevent numerical issues in loss)
    model.add(layers.Dense(1, activation='linear', dtype='float32'))

    # Select optimizer
    if solver.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate_init)
    elif solver.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate_init)
    else:
        optimizer = solver
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

class MLPKerasModel(BaseModel):
    """
    A pure Keras-based MLP regressor.
    """
    def __init__(self, hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', 
                 alpha=0.001, learning_rate_init=0.001, max_iter=2000, input_shape=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter  # Number of epochs
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        self.model = build_keras_mlp(hidden_layer_sizes=self.hidden_layer_sizes,
                                     activation=self.activation,
                                     solver=self.solver,
                                     alpha=self.alpha,
                                     learning_rate_init=self.learning_rate_init,
                                     input_shape=self.input_shape)
        return self.model

    def train(self, X, y, batch_size=512, verbose=2, use_early_stopping=True, patience=5):
        if self.model is None:
            if self.input_shape is None:
                self.input_shape = (X.shape[1],)
            self.build_model()
            
        # Convert X and y to NumPy arrays with a specific dtype.
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Convert data to a tf.data.Dataset, with shuffling, batching, and prefetching.
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Optionally add early stopping callback.
        callbacks = []
        if use_early_stopping:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=patience, restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        history = self.model.fit(dataset, epochs=self.max_iter, verbose=verbose, callbacks=callbacks)
        return history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(X)

    def save(self, path):
        if self.model is None:
            raise ValueError("Model has not been trained.")
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)