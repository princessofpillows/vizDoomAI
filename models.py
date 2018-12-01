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

def alexnet(cfg, num_classes):
    shape = (227, 227)
    model = tf.keras.Sequential([
        # Filters, Kernel Size, Strides
        tf.keras.layers.Conv2D(96, 11, 4, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(3, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 5, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(3, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(384, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(384, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(3, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(4096, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(num_classes, kernel_initializer=cfg.init)
    ])
    return model, shape

def zfnet(cfg, num_classes):
    shape =  (225, 225)
    model = tf.keras.Sequential([
        # Filters, Kernel Size, Strides
        tf.keras.layers.Conv2D(96, 7, 2, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 5, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(3, 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(1024, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(3, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(4096, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(num_classes, kernel_initializer=cfg.init)
    ])
    return model, shape

def vggnet(cfg, num_classes):
    shape = (224, 224)
    model = tf.keras.Sequential([
        # Filters, Kernel Size, Strides
        tf.keras.layers.Conv2D(64, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(4096, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Dense(num_classes, kernel_initializer=cfg.init)
    ])
    return model, shape

def googlenet(cfg, num_classes):
    shape = (224, 224)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 7, 2, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.MaxPool2D(3, 2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 1, activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.Conv2D(192, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(3, 2, padding="same"),
        Inception(cfg, 64, 96, 128, 16, 32, 32),
        Inception(cfg, 128, 128, 192, 32, 96, 64),
        tf.keras.layers.MaxPool2D(3, 2, padding="same"),
        Inception(cfg, 192, 96, 208, 16, 48, 64),
        # Record loss from auxiliary layer (disabled during inference time)
        #Auxiliary(cfg),
        Inception(cfg, 160, 112, 224, 24, 64, 64),
        Inception(cfg, 128, 128, 256, 24, 64, 64),
        Inception(cfg, 112, 144, 288, 32, 64, 64),
        # Record loss from auxiliary layer (disabled during inference time)
        #Auxiliary(cfg),
        Inception(cfg, 256, 160, 320, 32, 128, 128),
        tf.keras.layers.MaxPool2D(3, 2, padding="same"),
        Inception(cfg, 256, 160, 320, 32, 128, 128),
        Inception(cfg, 384, 192, 384, 48, 128, 128),
        tf.keras.layers.AveragePooling2D(7, 1),
        # Perform dropout to reduce overfitting
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, kernel_initializer=cfg.init)
    ])
    return model, shape
