import argparse
import matplotlib.pyplot as plt


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

        
def plot_magnitude_spectrum(magnitudes, 
                            title='', 
                            figure_size=(12, 8),
                            save_path=''):
    if save_path == '':
        raise ValueError('You need to pass a path to save the magnitude plot.')
    
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    
    if title != '':
        ax.set_title(title)
        
    plt.imshow(magnitudes)
    fig.savefig(save_path)
    plt.close(fig) 