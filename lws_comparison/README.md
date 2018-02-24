# Seq2Seq Audio

## Sample usage

One way you might train and run the model is like this:
```bash
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path lws_test \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1 
```

You can run more than one phase estimation method on the same model. Ensure all of the ancillary arguments are set for lws and griffin lim respectively.
```bash
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path phase_comparison \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 0
```

Run a sanity check to test I/O and signal reconstruction capacity.
```bash
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path sanity_check \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 0 \
    --sanity_check
```