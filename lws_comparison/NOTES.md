## To Do:
* Train and time all combinations of LSTM and Seq2Seq settings:
    - mags:           lws, librosa
    - hop size:       25% frame size
    - frame size:     512, 1024, 2048, 4096
    - perfect rec:    True, False
    - reconstruction: lws_music, lws_speech, griffin lim


## Notes:

* Difference between librosa and lws stfts:
```
# Get data.
data, sr = librosa.load(path)

# Get stfts from librosa and then the lws module.
lr_stfts = librosa.stft(data, n_fft=1024, hop_length=256)
lws_processor = lws.lws(1024, 256, mode='music', perfectrec=False)
lws_stfts = lws_processor.stft(data)

# Resultant code will return something like this!
# Different stft sequence length and different complex precision.
# If perfectrec is True then the amount of frames will be 11406.
# lr_stfts.shape  == (513, 11403)
# lr_stfts.dtype  == dtype('complex64')
# lws_stfts.shape == (513, 11400)
# lws_stfts.dtype == dtype('complex128')
```
