import argparse


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


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
