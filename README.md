# Mind-Cloud-AIC
Automatic Speech Recognition (ASR) System for Egyptian Arabic

## Dataset

The dataset comprises 50,715 audio files in the `/train/` folder and 2,199 audio files in the `/adapt/` folder. Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 16,000 Hz. The corresponding transcripts are provided in the `train.csv` file with the following fields:

- **audio**: The name of the corresponding `.wav` file.
- **Transcription**: The words spoken by the reader.

## Preprocessing

We prepared the vocabulary consisting of 38 characters, including the space and the OOV (out-of-vocabulary) token. The following transformations were applied to the data:

- **Spectrogram Generation**: Spectrograms were obtained using Short-Time Fourier Transform (STFT) with a frame length of 240 (15 ms), frame step of 120, and FFT length of 256.
- **Normalization**: Spectrograms were normalized.
- **Label Encoding**: Labels were split and encoded.
- **Batching**: A batch size of 32 was chosen.

![Example of the Spectrogram](https://github.com/Yahia-Ibrahim/mind-cloud-AIC/assets/120991373/5f83164f-89a3-4375-8392-1ae68542d696)

## Model Architecture

### Input Layer
- Accepts spectrogram inputs of shape `(None, input_dim)`.

### Convolutional Layers
- **Conv1**: 96 filters, kernel size `[11, 41]`, strides `[2, 2]`, followed by ReLU activation.
- **Conv2**: 128 filters, kernel size `[11, 21]`, strides `[1, 2]`, followed by ReLU activation.

### Reshape Layer
- Reshapes the output from the convolutional layers to a 2D tensor for the RNN layers.

### RNN Layers
- **Bidirectional GRU Layers**: Five layers, each with 768 units, to capture temporal dependencies. Each GRU layer uses `tanh` activation and `sigmoid` recurrent activation. Outputs from forward and backward GRU cells are concatenated.

### Dense Layer
- A fully connected layer with `2 * rnn_units` units followed by ReLU activation.

### Output Layer
- A dense layer with `output_dim + 1` units and a softmax activation function to predict character probabilities.

## Compilation

- **Optimizer**: Adam optimizer.
- **Loss Function**: Connectionist Temporal Classification (CTC) Loss.

## Summary

This model leverages Convolutional Neural Networks (CNNs) for feature extraction and Recurrent Neural Networks (RNNs) for sequence modeling, making it well-suited for end-to-end speech recognition tasks. The architecture ensures effective learning of both local and temporal features from spectrogram inputs.

You can download the model weights from [here](https://drive.google.com/file/d/1kW7POZ_S4dI9ixqYswXXqOGkFQzKi1WI/view?usp=drive_link).

## References 

- [Automatic Speech Recognition using CTC](https://keras.io/examples/audio/ctc_asr/)
- [Multi-Dialect Arabic Speech Recognition](https://arxiv.org/pdf/2112.14678)
