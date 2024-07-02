# mind-cloud-AIC
ASR system for egyption arabic
## Load the Dataset

first we downloaded the dataset.

The dataset contains 50,715 audio files as `wav` files in the `/train/` folder, and 2,199 audio files in the `/adapt/` folder.
The label (transcript) for each audio file is a string
given in the `train.csv` file. The fields are:

- **audio**: this is the name of the corresponding .wav file
- **Transcription**: words spoken by the reader.

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 16,000 Hz.


## Preprocessing

We first prepared the vocabulary to be used which are 38 charchters contaning the space and the oov token.

then we applied the following transformations:

- spectrograms of the data were optained using stft transfromation using frame length of 240 (corresponding to 15 ms), a frame step of 120 and fft length of 256.
- spectrograms of the whole data were normalized.
- the labels were split and encoded.
- batch size of 32 was chosen.

![example of the spectrogram](https://github.com/Yahia-Ibrahim/mind-cloud-AIC/assets/120991373/5f83164f-89a3-4375-8392-1ae68542d696)


### Model Architecture

- **Input Layer:** 
  - Accepts spectrogram inputs of shape.
  
- **Convolutional Layers:**
  - Two 2D convolutional layers are used to capture local patterns in the spectrogram:
    - **Conv1:** 96 filters, kernel size `[11, 41]`, strides `[2, 2]`, followed by ReLU activation.
    - **Conv2:** 128 filters, kernel size `[11, 21]`, strides `[1, 2]`, followed by ReLU activation.

- **Reshape Layer:**
  - The output from the convolutional layers is reshaped to a 2D tensor for the RNN layers.

- **RNN Layers:**
  - Five Bidirectional GRU layers, each with 768 units, are used to capture temporal dependencies:
    - Each GRU layer uses `tanh` activation and `sigmoid` recurrent activation.
    - Bidirectional GRU layers concatenate the outputs of forward and backward GRU cells.

- **Dense Layer:**
  - A fully connected layer with `2 * rnn_units` units followed by ReLU activation.
  
- **Output Layer:**
  - The final output layer is a dense layer with `output_dim + 1` units and a softmax activation function to predict character probabilities.

### Compilation

- **Optimizer:** Adam optimizer.
- **Loss Function:** CTC (Connectionist Temporal Classification) Loss.

### Summary

This model leverages CNNs for feature extraction and RNNs for sequence modeling, making it well-suited for end-to-end speech recognition tasks. The model's architecture ensures effective learning of both local and temporal features from spectrogram inputs.
