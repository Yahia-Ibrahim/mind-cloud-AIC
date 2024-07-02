# mind-cloud-AIC
ASR system for egyption arabic
## Load the LJSpeech Dataset

first we downloaded the dataset.

The dataset contains 50,715 audio files as `wav` files in the `/train/` folder, and 2,199 audio files in the `/adapt/` folder.
The label (transcript) for each audio file is a string
given in the `train.csv` file. The fields are:

- **audio**: this is the name of the corresponding .wav file
- **Transcription**: words spoken by the reader.

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 16,000 Hz.
