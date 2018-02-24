# Seq2Seq Audio

## Sample usage

One way you might train and run the model is like this:
```bash
python main.py \
    --epochs 20 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path ./saves \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --lws_mags 1 \
    --phase_estimation_method lws \
    --perfect_reconstruction 1 \
    --lws_mode music \
    --cuda_device 1 
```