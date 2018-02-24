import argparse


def get_file_name(settings):

    file_name = "fft_size_{}_lws_mags_{}_gl_phase_{}"

    if settings.griffin_lim_phase == 1:
        file_name += "_gl_iters_{}_overlap_{}.wav"
        file_name = file_name.format(settings.fft_size,
                                     int(settings.lws_mags),
                                     int(settings.griffin_lim_phase),
                                     settings.griffin_lim_iterations,
                                     settings.overlap_ratio)
    else:
        file_name += "_perfect_rec_{}_mode_{}_overlap_{}.wav"
        file_name = file_name.format(settings.fft_size,
                                     int(settings.lws_mags),
                                     int(settings.griffin_lim_phase),
                                     int(settings.perfect_reconstruction),
                                     settings.mode,
                                     settings.overlap_ratio)
    return file_name


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
