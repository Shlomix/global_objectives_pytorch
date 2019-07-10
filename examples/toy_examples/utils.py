import numpy as np
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt


def create_training_and_eval_data_for_experiment(**data_config):

  def data_points(is_positives, index):
    variance = data_config['positives_variances'
                           if is_positives else 'negatives_variances'][index]
    center = data_config['positives_centers'
                         if is_positives else 'negatives_centers'][index]
    count = data_config['positives_counts'
                        if is_positives else 'negatives_counts'][index]
    return variance*np.random.randn(count, 2) + np.array([center])

  def create_data():
    return np.concatenate([data_points(False, 0),
                           data_points(True, 0),
                           data_points(True, 1),
                           data_points(False, 1)], axis=0)

  def create_labels():
    return np.array([0.0]*data_config['negatives_counts'][0] +
                    [1.0]*data_config['positives_counts'][0] +
                    [1.0]*data_config['positives_counts'][1] +
                    [0.0]*data_config['negatives_counts'][1])

  permutation = np.random.permutation(
      sum(data_config['positives_counts'] + data_config['negatives_counts']))

  train_data = create_data()[permutation, :]
  eval_data = create_data()[permutation, :]
  train_labels = create_labels()[permutation]
  eval_labels = create_labels()[permutation]

  return {
      'train_data': train_data,
      'train_labels': train_labels,
      'eval_data': eval_data,
      'eval_labels': eval_labels
  }


def true_positive_at_false_positive(scores, labels, target_fpr):
    negative_scores = scores[labels == 0.0]
    threshold = np.percentile(negative_scores, 100 - 100 * target_fpr)
    predicted = scores >= threshold
    return recall_score(labels, predicted), \
           (negative_scores >= threshold).sum() / negative_scores.size, \
           threshold


def precision_at_recall(scores, labels, target_recall):
    positive_scores = scores[labels == 1.0]
    threshold = np.percentile(positive_scores, 100 - target_recall*100)
    predicted = scores >= threshold
    return precision_score(labels, predicted), \
           recall_score(labels, predicted), \
           threshold


def plot_results(data,
                 w_1, b_1, threshold,
                 w_2, b_2,
                 obj_type, at_target_type,
                 at_target_rate):

    labels = ["negative_train", "positive_train", "negative_test", "positive_test"]
    colors = ["navy", "firebrick", "blue", "red"]
    markers = ['-', '+', '-', '+']

    x = dict()
    x[labels[0]] = data["train_data"][data["train_labels"] == 0, :]
    x[labels[1]] = data["train_data"][data["train_labels"] == 1, :]
    x[labels[2]] = data["eval_data"][data["eval_labels"] == 0, :]
    x[labels[3]] = data["eval_data"][data["eval_labels"] == 1, :]

    for l, c, m in zip(labels, colors, markers):
        plt.scatter(x[l][:,0], x[l][:, 1],
                    color=c, label=l
                    )

    xx = np.linspace(-2., 2., 30)

    plt.plot(xx, -xx*w_1[0]/w_1[1] - b_1/w_1[1],
             color='purple', label="Cross Entropy"
             )

    plt.plot(xx, -xx*w_1[0]/w_1[1] - (b_1 - threshold)/w_1[1],
             color='green', label="Cross Entropy @ {}{}".
             format(at_target_type, at_target_rate)
             )
    plt.plot(xx, -xx*w_2[0]/w_2[1] - b_2/w_2[1],
             color='k', label="{}@{}{}".
             format(obj_type, at_target_type, at_target_rate)
             )
    plt.xlim(-1, 1.7)
    plt.ylim(-1.5, 2)
    plt.grid()
    plt.legend()
    plt.show()