import os, sys

path = os.path.abspath(__file__ + "/../../../")
sys.path.insert(0, path)

from global_objectives.losses import PRLoss
from examples.toy_examples.utils import *
from examples.toy_examples.trainer import train_model


TARGET_RECALL = 0.98
TRAIN_ITERATIONS = 3000
LEARNING_RATE = 0.01
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


def main(unused_argv):
    del unused_argv
    experiment_data = create_training_and_eval_data_for_experiment(
        **EXPERIMENT_DATA_CONFIG)

    print('### Training with cross_entropy loss:')

    p_1, r_1, w_1, b_1, threshold = train_model(data=experiment_data,
                                         use_global_objectives=False,
                                         metric_func=precision_at_recall,
                                         at_target_rate=TARGET_RECALL,
                                         obj_type='P', at_target_type='R',
                                         train_iteration=TRAIN_ITERATIONS,
                                         lr=LEARNING_RATE,
                                         num_checkpoints=NUM_CHECKPOINTS)

    print('cross_entropy_loss precision at requested recall '
          'is {:.2f}@{:.2f}\n'.format(p_1, r_1))

    criterion = PRLoss(target_recall=TARGET_RECALL, num_labels=1,
                       dual_factor=1.0)

    print('\n\n### training precision@recall loss:')

    p_2, r_2, w_2, b_2, _ = train_model(data=experiment_data,
                                         use_global_objectives=True,
                                         criterion=criterion,
                                         metric_func=precision_at_recall,
                                         at_target_rate=TARGET_RECALL,
                                         obj_type='TPR', at_target_type='FPR',
                                         train_iteration=TRAIN_ITERATIONS,
                                         lr=LEARNING_RATE,
                                         num_checkpoints=NUM_CHECKPOINTS)


    print('precision_at_recall_loss precision at requested recall '
          'is {:.2f}@{:.2f}'.format(p_2, r_2))

    plot_results(data=experiment_data,
                 w_1=w_1, b_1=b_1,
                 threshold=threshold,
                 w_2=w_2, b_2=b_2,
                 obj_type="P",
                 at_target_type="R",
                 at_target_rate=TARGET_RECALL)


if __name__ == '__main__':
    main("run")
