import os
import tensorflow as tf
print(tf.__version__)
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"