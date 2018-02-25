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


## Sanity check output
If file begins with estimated, then it the phase has been estimated using either lws, vocoder or griffinlim. The file begins with recovered, the original phases were used. 

If the the phase has been estimated, there are three words after 'estimated'. The first denotes the phase estimation method, the second denotes the stft and istft library (lws or librosa), and the final words denote the audio file name. For example:
```
estimated_griffin_lim_librosa_test_piano.wav
```
Is an example of the phase being estimated by griffin lim using the librosa stft and istft functions on the file `test_piano.wav`.

If the phase has been recovered from the original phase, the word after recovered will denote the stft and istft library used, either librosa or lws.


```bash
*******************************************************
NEW FILE: test_sweep.wav
*******************************************************

*******************************************************
mags + original phase result:
    * file: test_sweep.wav,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 2.385726927134405e-06
*******************************************************

mags + estimated phase result:
    * file: test_sweep.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 0.23671948505851334

mags + estimated phase result:
    * file: test_sweep.wav,
    * phase estimation: lws,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 3.2711855016122717

mags + estimated phase result:
    * file: test_sweep.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 2.7769642180321745

*******************************************************
mags + original phase result:
    * file: test_sweep.wav,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 2.4101797238114133
*******************************************************

mags + estimated phase result:
    * file: test_sweep.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 3.230904429539939

mags + estimated phase result:
    * file: test_sweep.wav,
    * phase estimation: lws,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 3.4873112432170776

mags + estimated phase result:
    * file: test_sweep.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 4.823160907373881

*******************************************************
NEW FILE: test_bach.wav
*******************************************************

*******************************************************
mags + original phase result:
    * file: test_bach.wav,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 4.861757368509972e-06
*******************************************************

mags + estimated phase result:
    * file: test_bach.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 0.438362727610813

mags + estimated phase result:
    * file: test_bach.wav,
    * phase estimation: lws,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 4.079438280536972

mags + estimated phase result:
    * file: test_bach.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 6.278623207977225

*******************************************************
mags + original phase result:
    * file: test_bach.wav,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 7.276213218909443
*******************************************************

mags + estimated phase result:
    * file: test_bach.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 3.7721790322960658

mags + estimated phase result:
    * file: test_bach.wav,
    * phase estimation: lws,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 7.276966528387836

mags + estimated phase result:
    * file: test_bach.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 8.654241642907204

*******************************************************
NEW FILE: test_noise.wav
*******************************************************

*******************************************************
mags + original phase result:
    * file: test_noise.wav,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 4.1629345126439387e-07
*******************************************************

mags + estimated phase result:
    * file: test_noise.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 0.4450537526565462

mags + estimated phase result:
    * file: test_noise.wav,
    * phase estimation: lws,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 3.6120162824129225

mags + estimated phase result:
    * file: test_noise.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 5.845272569184445

*******************************************************
mags + original phase result:
    * file: test_noise.wav,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 3.3402473412770415
*******************************************************

mags + estimated phase result:
    * file: test_noise.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 3.0021743032749244

mags + estimated phase result:
    * file: test_noise.wav,
    * phase estimation: lws,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 3.341855677652547

mags + estimated phase result:
    * file: test_noise.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 6.902189439097297

*******************************************************
NEW FILE: test_piano.wav
*******************************************************

*******************************************************
mags + original phase result:
    * file: test_piano.wav,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 5.236772177933148e-06
*******************************************************

mags + estimated phase result:
    * file: test_piano.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 0.4441820880216123

mags + estimated phase result:
    * file: test_piano.wav,
    * phase estimation: lws,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 5.098599938859174

mags + estimated phase result:
    * file: test_piano.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: librosa,
    * mean mfcc distance: 6.337462858019892

*******************************************************
mags + original phase result:
    * file: test_piano.wav,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 4.111474145103629
*******************************************************

mags + estimated phase result:
    * file: test_piano.wav,
    * phase estimation: griffin_lim,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 2.5632558530060097

mags + estimated phase result:
    * file: test_piano.wav,
    * phase estimation: lws,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 4.089570474893734

mags + estimated phase result:
    * file: test_piano.wav,
    * phase estimation: vocoder,
    * magnitude source / stft library: lws,
    * mean mfcc distance: 6.996910963720909
```