import os, sys

path = os.path.abspath(__file__ + "/../../../")
sys.path.insert(0, path)

from global_objectives.losses import AUCROCLoss
from examples.toy_examples.utils import *
from examples.toy_examples.trainer import train_model


TARGET_FPR = 0.01
TRAIN_ITERATIONS = 4000
LEARNING_RATE = 0.1
NUM_CHECKPOINTS = 10


EXPERIMENT_DATA_CONFIG = {
    'positives_centers': [[0, 1.0], [1.0, -0.5]],
    'negatives_centers': [[0.0, -0.5], [1.0, 1.0]],
    'positives_variances': [0.15, 0.1],
    'negatives_variances': [0.15, 0.1],
    'positives_counts': [500, 100],
    'negatives_counts': [2000, 50]
}


def main(unused_argv):
    del unused_argv
    experiment_data = create_training_and_eval_data_for_experiment(
        **EXPERIMENT_DATA_CONFIG)

    tpr_1, fpr_1, w_1, b_1, threshold = train_model(
        data=experiment_data,
        use_global_objectives=False,
        metric_func=true_positive_at_false_positive,
        at_target_rate=TARGET_FPR,
        obj_type='TPR', at_target_type='FPR',
        train_iteration=TRAIN_ITERATIONS,
        lr=LEARNING_RATE,
        num_checkpoints=NUM_CHECKPOINTS
    )

    print('cross_entropy_loss tpr at requested fpr is {:.2f}@{:.2f}\n'.
          format(tpr_1, fpr_1)
          )

    criterion = AUCROCLoss(fp_range_lower=0.0, fp_range_upper=1.0, num_labels=1, num_anchors=5)

    tpr_2, fpr_2, w_2, b_2, _ = train_model(
        data=experiment_data,
        use_global_objectives=True,
        criterion=criterion,
        metric_func=true_positive_at_false_positive,
        at_target_rate=TARGET_FPR,
        obj_type='TPR', at_target_type='FPR',
        train_iteration=TRAIN_ITERATIONS,
        lr=LEARNING_RATE,
        num_checkpoints=NUM_CHECKPOINTS
    )

    print('true_positives_at_false_positives_loss tpr '
          'at requested fpr is {:.2f}@{:.2f}'.
          format(tpr_2, fpr_2)
          )

    plot_results(
        data=experiment_data,
        w_1=w_1, b_1=b_1,
        threshold=threshold,
        w_2=w_2, b_2=b_2,
        obj_type="TPR",
        at_target_type="FPR",
        at_target_rate=TARGET_FPR
    )


if __name__ == '__main__':
    main("run")
