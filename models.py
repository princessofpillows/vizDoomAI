import tensorflow as tf
from layers import Inception, Auxiliary


def atari(cfg, num_classes):
    shape = (84, 84)
    model = tf.keras.Sequential([
        # Filters, Kernel Size, Strides
        tf.keras.layers.Conv2D(16, 8, 4, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(32, 4, 2, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(len(cfg.actions), kernel_initializer=cfg.init)
    ])
    return model, shape
