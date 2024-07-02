# mind-cloud-AIC
ASR system for egyption arabic
## Load the LJSpeech Dataset

first we downloaded the dataset.
The dataset contains 13,100 audio files as `wav` files in the `/wavs/` folder.
The label (transcript) for each audio file is a string
given in the `metadata.csv` file. The fields are:

- **ID**: this is the name of the corresponding .wav file
- **Transcription**: words spoken by the reader.

Each audio file is a single-channel 16-bit PCM WAV with a sample rate of 16,000 Hz.
