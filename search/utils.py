
import numpy as np
import librosa

# For to_var.
import torch
from torch.autograd import Variable

# Stats for outlier detection.
from scipy import stats
from sklearn import decomposition
from sklearn.neighbors import LocalOutlierFactor

# To make nice plots.
import matplotlib.pyplot as plt


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def griffin_lim(spectrogram,
                n_iter=100,
                window='hann',
                n_fft=2048,
                hop_length=-1):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)
        rebuilt = librosa.stft(inverse, n_fft=n_fft,
                               hop_length=hop_length, window=window)
        angles = np.exp(1j * np.angle(rebuilt))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, window=window)

    return inverse


def plot_audio(data):
    plt.figure()
    stft = librosa.amplitude_to_db(librosa.stft(data), ref=np.max)
    librosa.display.specshow(stft, y_axis='log')
    plt.show()


def outlier_detection(data,
                      contamination_fraction=0.25,
                      amount_neighbours=35,
                      figure_width=8,
                      inline=False):

    # Get rank-2 data points.
    number_data_points = len(data)
    data = np.array(data).reshape((number_data_points, -1))

    # perform dimensionality reduction if necessary for visualisation.
    if data.shape[1] > 2:
        pca = decomposition.PCA(n_components=2)
        pca.fit(data)
        data = pca.transform(data)

    # Setup.
    xx, yy = np.meshgrid(np.linspace(data.min(), data.max(), 100),
                         np.linspace(data.min(), data.max(), 100))

    # Instanciate outlier detection method to find anomolies.
    lof = LocalOutlierFactor(n_neighbors=amount_neighbours,
                             contamination=contamination_fraction)
    lof_predictions = lof.fit_predict(data)
    lof_scores_pred = lof.negative_outlier_factor_
    lof_threshold = stats.scoreatpercentile(lof_scores_pred,
                                            100 * contamination_fraction)
    lof_z = lof._decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Plot everything.
    plot_data = [
        ('Local Outlier Factor', lof_z, lof_threshold, lof_predictions)
    ]

    fig = plt.figure(figsize=(figure_width, figure_width * len(plot_data)))
    plot_number = 1
    for title, z, threshold, predictions in plot_data:
        z = z.reshape(xx.shape)
        sub_plot = fig.add_subplot(len(plot_data), 1, plot_number)
        sub_plot.contourf(xx, yy, z, levels=np.linspace(
            z.min(), threshold, 7), cmap=plt.cm.Blues_r)
        sub_plot.contour(xx, yy, z, levels=[
                         threshold], linewidths=2, colors='red')
        sub_plot.contourf(xx, yy, z, levels=[
                          threshold, z.max()], colors='orange')

        inliers = data[np.where(predictions == 1)]
        outliers = data[np.where(predictions == -1)]
        sub_plot.scatter(inliers[:, 0], inliers[:, 1],
                         c='white', s=20, edgecolor='k')
        sub_plot.scatter(outliers[:, 0], outliers[:, 1],
                         c='black', s=20, edgecolor='k')

        sub_plot.axis('tight')
        sub_plot.set_xlabel(title)
        sub_plot.set_xlim((data.min(), data.max()))
        sub_plot.set_ylim((data.min(), data.max()))

        plot_number += 1

    # Convert to NumPy array and return!
    fig.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    fig.suptitle("Outlier detection")

    if not inline:
        plt.close()
    else:
        fig.savefig('outlier_detection.png', dpi=fig.dpi)

    return fig, np.where(lof_predictions == -1)[0]
