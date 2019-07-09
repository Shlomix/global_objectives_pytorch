# Copyright 2018 The TensorFlow Global Objectives Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example for using global objectives.

Illustrate, using synthetic data, how using the precision_at_recall loss
significanly improves the performace of a linear classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from sklearn.metrics import precision_score
import torch
import torch.optim as optim

from losses_new import PRLoss, AUCPRLoss

#import loss_layers

# When optimizing using global_objectives, if set to True then the saddle point
# optimization steps are performed internally by the Tensorflow optimizer,
# otherwise by dedicated saddle-point steps as part of the optimization loop.
#USE_GO_SADDLE_POINT_OPT = False

TARGET_RECALL = 0.98
TRAIN_ITERATIONS = 4000
LEARNING_RATE = 0.05
GO_DUAL_RATE_FACTOR = 15.0
NUM_CHECKPOINTS = 10


EXPERIMENT_DATA_CONFIG = {
    'positives_centers': [[0, 1.0], [1, -0.5]],
    'negatives_centers': [[0, -0.5], [1, 1.0]],
    'positives_variances': [0.15, 0.1],
    'negatives_variances': [0.15, 0.1],
    'positives_counts': [500, 50],
    'negatives_counts': [3000, 100]
}


def create_training_and_eval_data_for_experiment(**data_config):
  """Creates train and eval data sets.

  Note: The synthesized binary-labeled data is a mixture of four Gaussians - two
    positives and two negatives. The centers, variances, and sizes for each of
    the two positives and negatives mixtures are passed in the respective keys
    of data_config:

  Args:
      **data_config: Dictionary with Array entries as follows:
        positives_centers - float [2,2] two centers of positives data sets.
        negatives_centers - float [2,2] two centers of negatives data sets.
        positives_variances - float [2] Variances for the positives sets.
        negatives_variances - float [2] Variances for the negatives sets.
        positives_counts - int [2] Counts for each of the two positives sets.
        negatives_counts - int [2] Counts for each of the two negatives sets.

  Returns:
    A dictionary with two shuffled data sets created - one for training and one
    for eval. The dictionary keys are 'train_data', 'train_labels', 'eval_data',
    and 'eval_labels'. The data points are two-dimentional floats, and the
    labels are in {0,1}.
  """
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
    """Creates an array of 0.0 or 1.0 labels for the data_config batches."""
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


def train_model(data, use_global_objectives):

    """Trains a linear model for maximal accuracy or precision at given recall."""

    def precision_at_recall(scores, labels, target_recall):
        """Computes precision - at target recall - over data."""
        positive_scores = scores[labels == 1.0]
        threshold = np.percentile(positive_scores, 100 - target_recall*100)
        predicted = scores >= threshold
        return precision_score(labels, predicted)

    device = torch.device('cuda')

    w = torch.tensor([0.0, -1.0],  requires_grad=True, device=device)
    b = torch.tensor([0.0], requires_grad=True, device=device)

    x = torch.tensor(data['train_data'], requires_grad=False, device=device).float().cuda()
    labels = torch.tensor(data['train_labels'], requires_grad=False, device=device).float().cuda()

    params = [w,b]
    if use_global_objectives:
        criterion = PRLoss(num_classes=1, target_recall=TARGET_RECALL)
        params += list(criterion.parameters())
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    #criterion = PrecisionAtRecall(target_recall=TARGET_RECALL, num_classes=1)

    #params = [w,b] + list(criterion.parameters())
    optimizer = optim.SGD(params, lr=LEARNING_RATE)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)


    checkpoint_step = TRAIN_ITERATIONS // NUM_CHECKPOINTS


    for t in range(TRAIN_ITERATIONS):
        # Zero gradient at the start of the iteration
        optimizer.zero_grad()

        # Forward pass: Compute predicted y by passing x to the model
        logits = x.mv(w) + b

        # Compute and print loss
        loss = criterion(logits, labels)

        # perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()
        #scheduler.step()
        #print(t, loss.item())

        if t % checkpoint_step == 0:
            w_ = w.cpu().detach().numpy()
            b_ = b.cpu().detach().numpy()

            precision = precision_at_recall(
                np.dot(data['train_data'], w_) + b_,
                data['train_labels'], TARGET_RECALL)

            print('Loss = {} Precision = {}'.format(loss.data, precision))


    w_ = w.cpu().detach().numpy()
    b_ = b.cpu().detach().numpy()

    precision = precision_at_recall(
        np.dot(data['eval_data'], w_) + b_,
        data['eval_labels'], TARGET_RECALL)

    return precision, w_, b_


def main(unused_argv):
    del unused_argv
    experiment_data = create_training_and_eval_data_for_experiment(
        **EXPERIMENT_DATA_CONFIG)

    #precision_1, w1, b1 = train_model(experiment_data, use_global_objectives=False)
    #print('cross_entropy precision at requested recall is {}'.format(precision_1))

    precision_2, w2, b2 = train_model(experiment_data, use_global_objectives=True)
    print('cross_entropy precision at requested recall is {}'.format(precision_2))

"""""""""
    import matplotlib.pyplot as plt


    labels = ["negative_train", "positive_train", "negative_test", "positive_test"]
    colors = ["navy", "firebrick", "blue", "red"]
    markers = ['-', '+', '-', '+']

    x = {}
    x[labels[0]] = experiment_data["train_data"][experiment_data["train_labels"] == 0, :]
    x[labels[1]] = experiment_data["train_data"][experiment_data["train_labels"] == 1, :]
    x[labels[2]] = experiment_data["eval_data"][experiment_data["eval_labels"] == 0, :]
    x[labels[3]] = experiment_data["eval_data"][experiment_data["eval_labels"] == 1, :]

    for l, c, m in zip(labels, colors, markers):
        plt.scatter(x[l][:,0], x[l][:, 1], color=c, label=l)

    xx = np.linspace(-2., 2., 30)

    plt.plot(xx, -xx*w1[0]/w1[1] - b1/w1[1], color='g', label="Cross Entropy")
    plt.plot(xx, -xx*w2[0]/w2[1] - b2/w2[1], color='m', label="P@R 0.98")
    plt.xlim(-1, 1.7)
    plt.ylim(-1.5, 2)
    plt.grid()
    plt.legend()
    plt.show()

"""""""""

if __name__ == '__main__':
    main("run")