import argparse


def get_file_name(settings):
    file_name =  "fft_size_{}_lws_mags_{}"
    file_name += "_gl_iters_{}_perfect_rec_{}"
    file_name += "_mode_{}_overlap_{}"
    file_name = file_name.format(settings.fft_size, 
                                 int(settings.lws_mags),
                                 settings.griffin_lim_iterations,
                                 int(settings.perfect_reconstruction),
                                 settings.mode,
                                 settings.overlap_ratio)
    return file_name


def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
