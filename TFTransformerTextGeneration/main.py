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
# Set the directories where the training and test datasets are stored
train_dir = "train"
test_dir = "test"

# Set the labels
labels = ["other"]

# Initialize lists to store the input and target sequences
input_sequences = []
target_sequences = []

# Iterate over the labels
for label in labels:
  # Get the list of file names in the train and test directories for the current label
  train_files = os.listdir(os.path.join(train_dir, label))
  test_files = os.listdir(os.path.join(test_dir, label))

  # Iterate over the file names in the train and test directories
  for train_file, test_file in zip(train_files, test_files):
    # Read the input and target sequences from the files
    with open(os.path.join(train_dir, label, train_file), "r") as f:
      input_sequence = f.read()
    with open(os.path.join(test_dir, label, test_file), "r") as f:
      target_sequence = f.read()

    # Append the input and target sequences to the lists
    input_sequences.append(input_sequence)
    target_sequences.append(target_sequence)

# Tokenize the input and target sequences
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()

# Fit the Tokenizer objects on the input and target sequences
input_tokenizer.fit_on_texts(input_sequences)
target_tokenizer.fit_on_texts(target_sequences)

# Get the vocabulary size of the input and target sequences
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Tokenize the input and target sequences
input_sequences = input_tokenizer.texts_to_sequences(input_sequences)
target_sequences = target_tokenizer.texts_to_sequences(target_sequences)

# Pad the input and target sequences
max_length = max(len(input_sequence) for input_sequence in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding="post")
target_sequences = pad_sequences(target_sequences, maxlen=max_length, padding="post")


# Set the hyperparameters of the model

num_layers = 6
d_model = 512
num_heads = 8
dff = 2048
EPOCHS = 12
# Set the batch size and buffer size
BATCH_SIZE = 10
BUFFER_SIZE = 10000


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Create the Transformer model
model = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size)

# Define the loss function and the optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



# Define the metric for measuring the model's performance
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Define the checkpoint callback to save the model's weights
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Define the loss and accuracy history
loss_history = []
acc_history = []



def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask



# Define the dataset and the iterator
dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences)).batch(BATCH_SIZE)
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
input_text = "Hi"
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_text)
input_vocab_size = len(input_tokenizer.word_index) + 1
input_sequence = input_tokenizer.texts_to_sequences([input_text])
input_sequence = pad_sequences(input_sequences, maxlen=max_length, padding="post")

# Create the masks
enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(input_sequences, target_sequences)

# Make a prediction
predict, _ = model(input_sequence, target_sequences, training=False, enc_padding_mask=enc_padding_mask, look_ahead_mask=look_ahead_mask, dec_padding_mask=dec_padding_mask)

# Get the index of the predicted word for each time step
predictions = tf.argmax(predict, axis=-1)

# Convert the indices to words
predicted_text = target_tokenizer.sequences_to_texts(predictions.numpy())

output_text = " ".join(predicted_text)
print(output_text)

