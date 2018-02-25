import os
import librosa
import textwrap
import numpy as np
from audio import magnitudes_to_audio, polar_to_cartesian


def check_normalisation_methods():
    pass

    

def get_mean_mfcc_distance(audio_a, audio_b, sample_rate):
        
    audio_a = audio_a.reshape((-1))
    audio_b = audio_b.reshape((-1))
    
    max_length = min(len(audio_a), len(audio_b))
    mfcc_a = librosa.feature.mfcc(y=audio_a[:max_length],
                                  sr=sample_rate)
    mfcc_b = librosa.feature.mfcc(y=audio_b[:max_length],
                                  sr=sample_rate)
    return np.mean(np.abs(mfcc_a - mfcc_b))


def sanity_check(settings, dataset):
    
    # Load audio settings.
    test_folder = 'test_wavs/'
    load_mono = True
    
    # Hack to ensure ensure mags are reconstructed correctly.
    old_settings_lws_mags = settings.lws_mags
    
    # Load each file in the test folder.
    for file_name in os.listdir(test_folder):
        
        # Only load wavs.
        if file_name.endswith('.wav'):
            
            # Get path to file.
            file_path = os.path.join(test_folder, file_name)
            
            # Load file to mono audio at forced sample rate.
            audio, _ = librosa.load(file_path, 
                                    mono=load_mono, 
                                    sr=dataset.sample_rate)
            
            star_length = 55
            stars = '*' * star_length
            print('\n' + stars, 
                  '\nNEW FILE: {}\n'.format(file_name) + stars)
            
            # We are interested in testing both sources of 
            # magnitudes.
            for magnitude_source in ['librosa', 'lws']:
                
                if magnitude_source == 'librosa':
                    stfts = librosa.stft(audio,
                                         n_fft=settings.fft_size,
                                         hop_length=settings.hop_length)
                    magnitudes, phases = librosa.magphase(stfts.T)
                    settings.lws_mags = False
                    
                    recovered_stfts = magnitudes * phases # Bad!
                    
                    # recovered_stfts = polar_to_cartesian(magnitudes, phases)
                    
                    # Try reconstructing the sound with the original
                    # phases using the librosa istft.
                    recovered_audio = librosa.istft(recovered_stfts.T,
                                                    hop_length=settings.hop_length)
                
                elif magnitude_source == 'lws':
                    stfts = dataset.lws_processor.stft(audio)
                    magnitudes = np.abs(stfts)
                    settings.lws_mags = True
                    
                    # Try reconstructing the sound with the original
                    # phases using the lws istft.
                    phases = np.exp(1.j * np.angle(stfts))
                    recovered_stfts = magnitudes.astype(np.complex128) * phases
                    recovered_audio = dataset.lws_processor.istft(recovered_stfts)
                    
                
                # Get perceptual distance.
                mfcc_distance = get_mean_mfcc_distance(audio, 
                                                       recovered_audio,
                                                       dataset.sample_rate)
                # Print results.
                result_str = """
                *******************************************************
                mags + original phase result:
                    * file: {}, 
                    * magnitude source / stft library: {}, 
                    * mean mfcc distance: {}
                *******************************************************"""
                result_str = result_str.format(file_name,
                                               magnitude_source, 
                                               mfcc_distance)
                print(textwrap.dedent(result_str))    
                # Save recovered audio to disk.
                # If specified save path, save output there. 
                # Else save in working directory.
                save_name = 'recovered_{}_{}'.format(magnitude_source, 
                                                     file_name)
                if settings.save_path is not None:
                    save_name = os.path.join(settings.save_path, save_name)
                librosa.output.write_wav(save_name, 
                                         recovered_audio, 
                                         dataset.sample_rate)
                    
                # Get phase estimated audio.
                for estimation_method in ['griffin_lim', 'lws', 'vocoder']:
                    reconstructed_audio = magnitudes_to_audio(magnitudes, 
                                                              settings, 
                                                              dataset, 
                                                              estimation_method)
                    # Get perceptual distance.
                    mfcc_distance = get_mean_mfcc_distance(audio, 
                                                           reconstructed_audio,
                                                           dataset.sample_rate)
                    
                    # Print results.
                    result_str = """
                    mags + estimated phase result:
                        * file: {}, 
                        * phase estimation: {}, 
                        * magnitude source / stft library: {}, 
                        * mean mfcc distance: {}"""
                    result_str = result_str.format(file_name,
                                                   estimation_method, 
                                                   magnitude_source, 
                                                   mfcc_distance)
                    print(textwrap.dedent(result_str))

                    # Save reconstructed audio to disk.
                    # If specified save path, save output there. 
                    # Else save in working directory.
                    save_name = 'estimated_{}_{}_{}'.format(estimation_method, 
                                                            magnitude_source, 
                                                            file_name)
                    if settings.save_path is not None:
                        save_name = os.path.join(settings.save_path, save_name)
                    librosa.output.write_wav(save_name, 
                                             reconstructed_audio, 
                                             dataset.sample_rate)
                    
    # Return settings to prior state!
    settings.lws_mags = old_settings_lws_mags