import os
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import layers
from transformer import Transformer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import seaborn as sns
# Tokenize the input and target sequences
input_text = ["Hola", "Como"]

target_text = ["Mundo", "Estas"]



# Create a Tokenizer object for the input and target sequences
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()

# Fit the Tokenizer objects on the input and target sequences
input_tokenizer.fit_on_texts(input_text)
target_tokenizer.fit_on_texts(target_text)

# Get the vocabulary size of the input and target sequences
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Tokenize the input and target sequences
input_sequence = input_tokenizer.texts_to_sequences([input_text])
target_sequence = target_tokenizer.texts_to_sequences([target_text])

# Pad the input and target sequences
max_length = max(len(input_sequence[0]), len(target_sequence[0]))
input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding="post")
target_sequence = pad_sequences(target_sequence, maxlen=max_length, padding="post")

# Set the hyperparameters of the model
num_layers = 6
d_model = 256
num_heads = 8
dff = 2048
EPOCHS = 6
# Set the batch size and buffer size
BATCH_SIZE = 32
BUFFER_SIZE = 1000

# Create the Transformer model
model = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size)

# Define the loss function and the optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the metric for measuring the model's performance
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Define the checkpoint callback to save the model's weights
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Define the loss and accuracy history
loss_history = []
acc_history = []

def create_masks(inputs, targets):
  # Encoder padding mask
  enc_padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)

  # Decoder padding mask
  dec_padding_mask = tf.cast(tf.math.equal(targets, 0), tf.float32)

  # Look ahead mask
  look_ahead_mask = tf.cast(tf.linalg.band_part(tf.ones((tf.shape(targets)[1], tf.shape(targets)[1])), -1, 0), tf.float32)

  return enc_padding_mask, look_ahead_mask, dec_padding_mask




# Define the dataset and the iterator
dataset = tf.data.Dataset.from_tensor_slices((input_sequence, target_sequence)).batch(BATCH_SIZE)
iterator = iter(dataset)

# Training loop
for epoch in range(EPOCHS):
  start = time.time()

  # Initialize the metric for each epoch
  accuracy.reset_states()

  # Iterate over the training data in batches
  for (batch, (inputs, targets)) in enumerate(dataset):
    # Generate the masks
    enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inputs, targets)

    # Make a forward pass through the model
    with tf.GradientTape() as tape:
      logits, _ = model(inputs, targets, True, enc_padding_mask, look_ahead_mask, dec_padding_mask)
      loss = loss_object(targets, logits)

    # Compute the gradients and apply them to the model's weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update the metric
    accuracy(targets, logits)

  # Log the loss and accuracy for each epoch
  loss_history.append(loss.numpy())
  acc_history.append(accuracy.result().numpy())
  print("Epoch {}: loss = {}, accuracy = {}".format(epoch+1, loss.numpy(), accuracy.result().numpy()))

  # Save the model's weights
  ckpt_manager.save()

  # Print the elapsed time for each epoch
  print("Time taken for epoch {}: {} secs\n".format(epoch+1, time.time() - start))

# Predict Hello World
# Load the saved model weights
ckpt.restore(ckpt_manager.latest_checkpoint)
# Predict Hello World
input_text = "Hola"
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts([input_text])
input_vocab_size = len(input_tokenizer.word_index) + 1
input_sequence = input_tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding="post")

# Create the masks
enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(input_sequence, target_sequence)

# Make a prediction
predict, _ = model(input_sequence, target_sequence, training=False, enc_padding_mask=enc_padding_mask, look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask)

# Get the index of the predicted word for each time step
predictions = tf.argmax(predict, axis=-1)

# Convert the indices to words
predicted_text = target_tokenizer.sequences_to_texts(predictions.numpy())

output_text = " ".join(predicted_text)
print(output_text)

