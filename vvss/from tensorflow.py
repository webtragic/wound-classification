from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

NUM_CLASSES = 3
IMG_SIZE = 224

# Load EfficientNetB0 model without top classification layers
base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Create a custom function to replace Swish activations with ReLU
def replace_swish_with_relu(layer):
    if isinstance(layer, tf.keras.layers.Activation) and 'swish' in layer.name:
        return layers.Activation('relu', name=layer.name.replace('swish', 'relu'))
    return layer

# Apply the custom function to the base model's layers
modified_layers = [replace_swish_with_relu(layer) for layer in base_model.layers]

# Create a new model with modified layers
model = tf.keras.Sequential(modified_layers)

# Add your own classification layers on top
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()
