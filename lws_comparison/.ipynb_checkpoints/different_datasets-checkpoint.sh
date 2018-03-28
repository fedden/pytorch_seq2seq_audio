#!/bin/sh

python -W ignore main.py \
    --verbose \
    --augmentions \
    --epochs 20 \
    --audio_path audio_datasets/drums_original.wav \
    --save_path augmented_drums_original \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1
    
python -W ignore main.py \
    --verbose \
    --augmentions \
    --epochs 20 \
    --audio_path audio_datasets/piano_original.wav \
    --save_path augmented_piano_original \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1   
    
python -W ignore main.py \
    --verbose \
    --augmentions \
    --epochs 20 \
    --audio_path audio_datasets/piano_trimmed.wav \
    --save_path augmented_piano_trimmed \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1
    
python -W ignore main.py \
    --verbose \
    --augmentions \
    --epochs 20 \
    --audio_path audio_datasets/vocals_original.wav \
    --save_path augmented_vocals_original \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1
    
python -W ignore main.py \
    --verbose \
    --augmentions \
    --epochs 20 \
    --audio_path audio_datasets/vocals_trimmed.wav \
    --save_path augmented_vocals_trimmed \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1















python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path audio_datasets/drums_original.wav \
    --save_path clean_drums_original \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1
    
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path audio_datasets/piano_original.wav \
    --save_path clean_piano_original \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1   
    
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path audio_datasets/piano_trimmed.wav \
    --save_path clean_piano_trimmed \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1
    
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path audio_datasets/vocals_original.wav \
    --save_path clean_vocals_original \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1
    
python -W ignore main.py \
    --verbose \
    --epochs 20 \
    --audio_path audio_datasets/vocals_trimmed.wav \
    --save_path clean_vocals_trimmed \
    --overlap_ratio 0.0 \
    --fft_size 2048 \
    --use_lws_mags True \
    --phase_estimation_methods lws griffin_lim vocoder \
    --griffin_lim_iterations 100 \
    --perfect_reconstruction True \
    --lws_mode music \
    --cuda_device 1






