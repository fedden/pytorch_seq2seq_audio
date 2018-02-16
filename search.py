import torch
import torch.nn as nn
from search.alchemist import Alchemist
from search.neural_functions import experiment

save_path = "experiment_22"
audio_path = "notebooks/massive_chops/trimmed/vocals_trimmed.wav"
population_size = 100
cuda_device = 1

with torch.cuda.device(cuda_device):

    manager = Alchemist(save_path=save_path,
                        search_type='random',
                        population_size=population_size)
    manager.add_experiment_fn(experiment_fn=experiment,
                              attention_method=['general', 'dot'],
                              fft_size=[128, 256, 512, 1024, 2048],
                              number_layers=[1, 2, 3],
                              dropout=[0.0, 0.1, 0.3, 0.6, 0.9],
                              sequence_length=[10, 30, 50, 70, 100, 150],
                              clip_threshold=(1.0, 50.0),
                              criterion=[nn.MSELoss(size_average=True),
                                         nn.MSELoss(size_average=False)],
                              learning_rate=(0.00005, 0.01),
                              decoder_learning_ratio=(1.0, 10.0),
                              batch_size=[8],
                              number_epochs=[20],
                              url=[None],
                              path=[audio_path],
                              input_noise=(0.0, 1.0))
    manager.run()
