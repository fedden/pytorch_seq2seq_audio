#!/bin/sh

python -W ignore main.py \
    --verbose \
    --epochs 13 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path lws_speed/vocals_2048_perfect_0 \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction False \
    --lws_mode music \
    --cuda_device 0 
    
python -W ignore main.py \
    --verbose \
    --epochs 13 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path lws_speed/vocals_1024_perfect_0 \
    --overlap_ratio 0.0 \
    --fft_size 1024 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction False \
    --lws_mode music \
    --cuda_device 0 

python -W ignore main.py \
    --verbose \
    --epochs 13 \
    --audio_path ../notebooks/massive_chops/trimmed/vocals_trimmed.wav \
    --save_path lws_speed/vocals_512_perfect_0 \
    --overlap_ratio 0.0 \
    --fft_size 512 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction False \
    --lws_mode music \
    --cuda_device 0 
