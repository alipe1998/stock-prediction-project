# src/models/mlp_keras.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, mixed_precision
from models.base_model import BaseModel

mixed_precision.set_global_policy("mixed_float16")

def build_keras_mlp(hidden_layer_sizes=(64, 32, 16), activation='relu',
                    solver='adam', alpha=0.001, learning_rate_init=0.001, input_shape=None):
    if input_shape is None:
        raise ValueError("input_shape must be provided for the model.")
    
    model = models.Sequential([tf.keras.Input(shape=input_shape)])
    
    for units in hidden_layer_sizes:
        model.add(layers.Dense(units,
                               activation=activation,
                               kernel_regularizer=regularizers.l2(alpha)))

    # Force output to float32 (important for TPU precision stability)
    model.add(layers.Dense(1, activation='linear', dtype='float32'))

    if solver.lower() == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate_init)
    elif solver.lower() == 'sgd':
        optimizer = optimizers.SGD(learning_rate=learning_rate_init)
    else:
        optimizer = solver

    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    return model

class MLPKerasModel(BaseModel):
    def __init__(self, hidden_layer_sizes=(64,32,16), activation='relu', solver='adam', 
                 alpha=0.001, learning_rate_init=0.001, max_iter=2000, input_shape=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.input_shape = input_shape
        self.model = None

        # TPU initialization
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            self.strategy = tf.distribute.TPUStrategy(tpu)
            print("Running on TPU:", tpu.master())
        except ValueError:
            self.strategy = tf.distribute.get_strategy()
            print("TPU not found, running on:", self.strategy)

    def build_model(self):
        with self.strategy.scope():
            self.model = build_keras_mlp(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                input_shape=self.input_shape
            )
        return self.model

    def train(self, X, y, batch_size=1024, verbose=2, use_early_stopping=True, patience=5):
        if self.model is None:
            if self.input_shape is None:
                self.input_shape = (X.shape[1],)
            self.build_model()

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        callbacks = []
        if use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=patience, restore_best_weights=True
            ))

        history = self.model.fit(
            dataset, epochs=self.max_iter, verbose=verbose, callbacks=callbacks
        )
        return history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained.")
        return self.model.predict(np.array(X, dtype=np.float32))

    def save(self, path):
        if self.model is None:
            raise ValueError("Model has not been trained.")
        self.model.save(path)

    def load(self, path):
        with self.strategy.scope():
            self.model = tf.keras.models.load_model(path)
