import os
import json
import librosa
import numpy as np
from inspect import signature
from collections import OrderedDict


def map_ranges(value, a1, a2, b1, b2):
    """A mapping function to map gene 0 - 1 to desired range."""
    return b1 + ((value - a1) * (b2 - b1) / (a2 - a1))


def get_selection(value, choices):
    """Get a value from the list or tuple."""
    if isinstance(choices, tuple):
        return map_ranges(value, 0, 1, choices[0], choices[1])
    elif isinstance(choices, list):
        index = int(map_ranges(value, 0, 1, 0, len(choices)))
        index = (index - 1) if index == len(choices) else index
        return choices[index]


def gene_to_arg_list(gene, options):
    """Turn gene into dict of arguments."""
    gene_str = "The gene and options should be proportionate"
    assert len(gene) == len(options), gene_str
    arguments = dict()
    i = 0
    for key, values in options.items():
        arguments[key] = get_selection(gene[i], values)
        i += 1
    return arguments


def audio_to_feature_vector(audio=None, sample_rate=None, path=None):
    """Convert audio at path or passed in to fixed size feature vector."""
    if path is not None:
        audio, sample_rate = librosa.load(path, mono=True)

    # Obtain the spectrogram.
    fft_size = 2048
    hop_size = fft_size // 4
    fft_frames = librosa.stft(y=audio, n_fft=fft_size, hop_length=hop_size)

    # Next get the MFCCs for timbre.
    mfccs = librosa.feature.mfcc(S=fft_frames, n_mfcc=13).T

    # Get the energies for loudness.
    rms_energies = librosa.feature.rmse(S=fft_frames).T

    # Get the spectral rolloff. Useful for voiced/unvoiced speech/music.
    rolloffs = librosa.feature.spectral_rolloff(y=audio,
                                                sr=sample_rate,
                                                n_fft=fft_size,
                                                hop_length=hop_size).T

    # Mean of each spectral frame.
    centroids = librosa.feature.spectral_centroid(y=audio,
                                                  sr=sample_rate,
                                                  n_fft=fft_size,
                                                  hop_length=hop_size).T
    # Concatenate all of the features together.
    desired_features = (mfccs, rms_energies, rolloffs, centroids)
    feature_vector = np.concatenate(desired_features, axis=1)

    # Cram it all into a single 1D vector of deviation, mean and first order
    # differences of the feature vector.
    feature_std = np.std(a=feature_vector, axis=0)
    feature_mean = np.mean(a=feature_vector, axis=0)
    feature_diff = np.mean(a=np.diff(a=feature_vector.T, n=1).T, axis=0)

    # Return the concatenation of these derivative features.
    return np.concatenate((feature_std, feature_mean, feature_diff))


def evaluate_fitness_novelty(results):
    pass


def evaluate_fitness_target(results, sample_rate, target):
    """Calculate average distance for each audio in the results batch."""
    vectors = [audio_to_feature_vector(audio=r, sample_rate=sample_rate)
               for r in results]
    distances = np.array([np.linalg.norm(v - target) for v in vectors])
    return np.mean(distances)


def genetic_algorithm(population, fitnesses, mutation_scale=0.01):
    """Calculate a round of genetic algorithm optimisation."""
    # Sanity conversion to numpy arrays.
    population, fitnesses = np.array(population), np.array(fitnesses)

    # Create fitness probability weightings.
    fitness_probability_weightings = fitnesses / fitnesses.sum()

    # Randomly get two parents indices for each new member of the population.
    parents_a_indices = np.random.choice(a=len(population),
                                         size=len(population),
                                         p=fitness_probability_weightings)
    parents_b_indices = np.random.choice(a=len(population),
                                         size=len(population),
                                         p=fitness_probability_weightings)

    # Get the parents.
    parents_a = population[parents_a_indices]
    parents_b = population[parents_b_indices]

    # Cross over the parents to make the children.
    crossover_probabilties = np.random.uniform(size=population.shape)
    children = np.where(crossover_probabilties < 0.5, parents_a, parents_b)

    # Mutate the children.
    mutation_noise = np.random.normal(scale=mutation_scale,
                                      size=population.shape)
    children += mutation_noise

    # Because the population is mapped to a set of ranges, ensure the children
    # are wrapped around one and zero.
    while children.max() > 1.0:
        children = np.where(children > 1.0, children - 1.0, children)
    while children.min() < 0.0:
        children = np.where(children < 0.0, children + 1.0, children)

    # Is every thing cool?
    child_str = "Children should have the same shape as the orignal population"
    assert children.shape == population.shape, child_str
    return children


def random_search(shape):
    return np.random.uniform(size=shape)


class Alchemist():

    """Alchemist class."""

    def __init__(self,
                 save_path='',
                 search_type='novelty',
                 target_path=None,
                 load_path=None,
                 population_size=100):

        # Sanity checks.
        search_str = "Search must be keyword."
        assert search_type in ['novelty', 'random', 'target'], search_str

        if search_str == 'target':
            target_str = "Please supply a target_path for the target vector."
            assert target_path is not None, target_str

        if load_path is not None:
            self.load_state()
        else:
            self.settings = dict()
            self.settings['experiment'] = 0
            self.settings['population_size'] = population_size
            self.population = None
            self.settings['search_type'] = search_type
            self.settings['save_path'] = save_path
            self.settings['mutation_scale'] = 0.01

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.train_fn = None
        self.save_fn = None
        self.visualise_fn = None
        self.experiment_options = None

        # Create target vector.
        if search_type == 'target':
            self.target = audio_to_feature_vector(path=target_path)

    def save_state(self):
        # Save population.
        population_path = os.path.join(self.save_path, 'population.npy')
        np.save(population_path, self.population)

        # Save settings.
        settings_path = os.path.join(self.save_path, 'settings.npy')
        with open(settings_path, 'w') as file_pointer:
            json.dump(self.settings, file_pointer, sort_keys=True, indent=4)

        if self.settings['search_type'] == 'target':
            np.save()

    def load_state(self):
        # Load the population.
        population_path = os.path.join(self.save_path, 'population.npy')
        np.save(population_path, self.population)

        # Load the settings.
        settings_path = os.path.join(self.save_path, 'settings.npy')
        with open(settings_path, 'r') as file_pointer:
            self.settings = json.load(file_pointer)

    def run(self):
        options_str = "run add_experiment_fn first!"
        assert self.experiment_options is not None, options_str

        # If beginning, create random population.
        if self.settings['experiment'] == 0 or self.population is None:
            gene_size = len(self.experiment_options)
            shape = (self.settings['population_size'], gene_size)
            self.population = random_search(shape)

        fitnesses = []

        # Then pass function to run experiment_fn to ga
        for gene in self.population:
            try:
                arguments = gene_to_arg_list(gene, self.experiment_options)
                results = self.experiment_fn(**arguments)

                if self.settings['search_type'] == 'novelty':
                    fitness = evaluate_fitness_novelty(results)
                elif self.settings['search_type'] == 'target':
                    sample_rate = self.experiment_options['sample_rate']
                    fitness = evaluate_fitness_target(results,
                                                      sample_rate,
                                                      self.target)

                fitnesses.append(fitness)
                self.settings['experiment'] += 1

            except (KeyboardInterrupt, SystemExit):
                raise

            except Exception as e:
                self.log_error(arguments, e)
                fitnesses.append(0.0)

        # Optimise population.
        if self.settings['search_type'] in ['novelty', 'target']:
            self.population = genetic_algorithm(self.population,
                                                fitnesses,
                                                self.mutation_scale)
        else:
            gene_size = len(self.experiment_options)
            shape = (self.population_size, gene_size)
            self.population = random_search(shape)

    def log_error(self, arguments, error):
        file_path = os.path.join(self.save_path, "log.txt")
        with open(file_path, "a") as log_file:
            line = "\nProgram settings '{}' produced '{}' as an error"
            error_str = str(error)
            settings = json.dumps(arguments, indent=4, sort_keys=True)
            log_file.write(line.format(settings, error_str))

    def add_experiment_fn(self, experiment_fn, **kwargs):
        amount_parameters = len(signature(experiment_fn).parameters)
        assert amount_parameters == len(kwargs.items()), \
            "Pass enough options for the experiment_fn's arguments"

        self.experiment_options = OrderedDict()
        for key, values in kwargs.items():

            if isinstance(values, tuple):
                assert len(values) == 2, "Pass a tuple of low/high values."
                self.experiment_options[key] = values

            elif isinstance(values, list):
                self.experiment_options[key] = values

        self.experiment_fn = experiment_fn
