# Import libraries
import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds 

# Function to put data into the same scale (numbers between 0 and 1)
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

# Load mnist from tensorflow dataset
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

# Split between train and test
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# Create samples to validate accuracy
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples 
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

# Create samples to test against real data
num_test_samples = 0.1 * mnist_info.splits['test'].num_examples 
num_test_samples = tf.cast(num_test_samples, tf.int64)

scaled_train_and_validation_data = mnist_train.map(scale)

test_data = mnist_test.map(scale)

BUFFER_SIZE = 10000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

# Create a keras model with two hidden layers

input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

NUM_EPOCHS = 5
VERBOSE = 2

model.fit(
    train_data,
    epochs = NUM_EPOCHS,
    validation_data=(validation_inputs, validation_targets),
    validation_steps=10,
    verbose=VERBOSE
)

# Evaluate with real data
test_loss, test_accuracy = model.evaluate(test_data)

print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%').format(test_loss, test_accuracy * 100.)

# 540/540 - 120s - loss: 0.0996 - accuracy: 0.9703 - val_loss: 0.1031 - val_accuracy: 0.9708

