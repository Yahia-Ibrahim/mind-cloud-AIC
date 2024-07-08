import subprocess
import sys
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", Path(__file__).parent / "requirements.txt"])

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
import os

# The set of characters accepted in the transcription.
characters = [x for x in "غظضذخثةتشقرصفعسمنلكيطحزوؤهدجبىائءإآأ "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# An integer scalar Tensor. The window length in samples.
frame_length = 240 
# An integer scalar Tensor. The number of samples to step.
frame_step = 120 
# An integer scalar Tensor. The size of the FFT to apply.
fft_length = 256 

def encode_single_sample_test(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(test_path + wav_file )
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    stfts = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )

    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(stfts)

    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ##  Process the label
    # 7. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    # 8. Map the characters in label to numbers
    label = char_to_num(label)
    # 9. Return a dict as our model is expecting two inputs
    return spectrogram, label   #spectrogram

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def build_model(input_dim, output_dim, rnn_layers= 5, rnn_units=256):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    
     # Convolutional layers
    for i, (filters, kernel_size, strides) in enumerate(
        [(96, [11, 41], [2, 2]), (128, [11, 21], [1, 2]) 
        ]
    ):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"conv_{i+1}",
            kernel_initializer=tf.initializers.GlorotUniform(),
        )(x)
        x = layers.ReLU(name=f"conv_{i+1}_relu")(x)

    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
            kernel_initializer=tf.initializers.GlorotUniform(),
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)


    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)

    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    
    opt = keras.optimizers.Adam(learning_rate=1e-8)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=768,
)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True, beam_width=512)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


model.load_weights("milestones/model_01_184.20.h5")

# Let's check results on more validation samples
test_path = "./test/"                 # assuming testing data are in the same directory
output_csv_path = 'Transcription.csv'
files = os.listdir(test_path)
empty_transcript = ""

df = pd.DataFrame({
    'audio': files,
    'transcript': [''] * len(files)  # Initialize the 'transcript' column with empty strings
})

batch_size = 32

test_dataset = tf.data.Dataset.from_tensor_slices(
    (np.array(df["audio"].tolist()), np.array(df["transcript"].tolist()))
)
test_dataset = (
    test_dataset.map(encode_single_sample_test, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

predictions = []
targets = []
for batch in test_dataset:
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)

new_df = pd.DataFrame({
    'audio': files,
    'transcript': predictions 
})
new_df['audio'] = new_df['audio'].str.replace('.wav', '', regex=False)

# Writing the DataFrame to a CSV file
new_df.to_csv('Transcription.csv', index=False, header=True)