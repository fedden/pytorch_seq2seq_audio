

def get_file_name(fft_size,
                  lws_mags,
                  griffin_lim_phase,
                  griffin_lim_iterations,
                  perfect_reconstruction,
                  mode):

    file_name = "fft_size_{}_lws_mags_{}_gl_phase_{}"
    if griffin_lim_phase == 1:
        file_name += "_gl_iters_{}.wav"
        file_name = file_name.format(fft_size,
                                     int(lws_mags),
                                     int(griffin_lim_phase),
                                     griffin_lim_iterations)
    else:
        file_name += "_perfect_rec_{}_mode_{}.wav"
        file_name = file_name.format(fft_size,
                                     int(lws_mags),
                                     int(griffin_lim_phase),
                                     int(perfect_reconstruction),
                                     mode)
    return file_name
