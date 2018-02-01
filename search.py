import torch.nn as nn
from search.alchemist import Alchemist
from search.neural_functions import experiment

save_path = "experiment_1"
audio_path = "/home/tollie/"
population_size = 100

manager = Alchemist(save_path=save_path,
                    search_type='novelty',
                    population_size=population_size)

manager.add_experiment_fn(experiment_fn=experiment,
                          attention_method=['general', 'dot', 'concat'],
                          fft_size=[128, 256, 512, 1024, 2048],
                          number_layers=[1, 2, 3, 4, 5, 6, 7],
                          dropout=[0.0, 0.2, 0.5, 0.8, 1.0],
                          sequence_length=[10, 30, 50, 70, 100, 150, 200],
                          clip_threshold=(1.0, 50.0),
                          criterion=[nn.MSELoss(size_average=True),
                                     nn.MSELoss(size_average=False),
                                     nn.L1Loss(size_average=True),
                                     nn.L1Loss(size_average=False),
                                     nn.SmoothL1Loss(size_average=True),
                                     nn.SmoothL1Loss(size_average=False)],
                          learning_rate=(0.00005, 0.01),
                          decoder_learning_ratio=(1.0, 10.0),
                          batch_size=[16, 32, 64],
                          number_epochs=[50, 75, 100, 125],
                          url=[None],
                          path=[audio_path])

manager.run()
