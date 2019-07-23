# Global Objectives For PyTorch
The Global Objectives library provides PyTorch loss layers that optimize
directly for a variety of objectives including true positive at false positive,
AUC, recall at precision, and more.

The global objectives losses can be used as drop-in replacements for
PyTorch's standard loss modules:
`torch.nn.CrossEntropy()` and `torch.nn.BCELoss()`.

Many machine learning classification models are optimized for classification
accuracy, when the real objective the user cares about is different and can be
precision at a fixed recall, precision-recall AUC, ROC AUC or similar metrics.
These are referred to as "global objectives" because they depend on how the
model classifies the dataset as a whole and do not decouple across data points
as accuracy does.

Because these objectives are combinatorial, discontinuous, and essentially
intractable to optimize directly, the functions in this library approximate
their corresponding objectives. This approximation approach follows the same
pattern as optimizing for accuracy, where a surrogate objective such as
cross-entropy or the hinge loss is used as an upper bound on the error rate.

## Getting Started
For a full example of how to use the loss functions in practice, see
examples/toy_examples/loss_layers_example.py.

Briefly, global objective losses can be used to replace
`torch.nn.BCEWithLogitsLoss()` by providing the relevant
additional arguments. For example,

``` python
torch.nn.BCELoss()
```

could be replaced with

``` python
global_objectives.PRLoss(
    target_recall=0.95)
```

Just as minimizing the cross-entropy loss will maximize accuracy, the loss
functions in losses.py were written so that minimizing the loss will
maximize the corresponding objective.


## Binary Label Format
Binary classification problems can be represented as a multi-class problem with
two classes, or as a multi-label problem with one label. (Recall that multiclass
problems have mutually exclusive classes, e.g. 'cat xor dog', and multilabel
have classes which are not mutually exclusive, e.g. an image can contain a cat,
a dog, both, or neither.) The softmax loss
(`torch.nn.CrossEntropy()`) is used for multi-class problems,
while the sigmoid loss (`torch.nn.BCEWithLogitsLoss()`) is used for
multi-label problems.

A multiclass label format for binary classification might represent positives
with the label [1, 0] and negatives with the label [0, 1], while the multilbel
format for the same problem would use [1] and [0], respectively.

All global objectives loss functions assume that the multilabel format is used.
Accordingly, if your current loss function is softmax, the labels will have to
be reformatted for the loss to work properly.

## Dual Variables
Global objectives losses use internal variables
called dual variables or Lagrange multipliers to enforce the desired constraint
(e.g. if optimzing for recall at precision, the constraint is on precision).

These dual variables are created and initialized internally by the loss
functions, and are updated during training by the same optimizer used for the
model's other variables. The dual variables can be found using the method 
the key `criterion.labmda_parameters()`.

## Loss Function Arguments
The following arguments are common to all loss functions in the library, and are
either required or very important.


* `num_labels (required)`: the number of labels must be direcly specified. 
* `dual_rate_factor (important)`: A floating point value which controls the step size for
  the Lagrange multipliers. Setting this value less than 1.0 will cause the
  constraint to be enforced more gradually and will result in more stable
  training.
* `num_anchors (important)` (AUCPRLoss and AUCROC only): The number of grid points used
   when approximating the AUC as a Riemann sum.
* `loss_type`: Either 'cross_entropy' (default) or 'hinge', specifying which upper bound 
   should be used for indicator functions.


In addition, the objectives with a single constraint (e.g.
`recall_at_precision_loss`) have an argument (e.g. `target_precision`) used to
specify the value of the constraint. The optional `precision_range` argument to
`precision_recall_auc_loss` is used to specify the range of precision values
over which to optimize the AUC, and defaults to the interval [0, 1].


## Forward() Method Arguments

The forward pass of the loss, e.g.: 
``` python
criterion = global_objectives.PRLoss(
    target_recall=0.95)
    ...
    ...
    ...
outputs = net(inputs)
loss = criterion(outputs, targets)
```
can/must receive the following arguments: 
* `labels (required)`: Corresponds directly to the `labels` argument of
  `torch.nn.BCEWithLogitsLoss()`.
* `logits (required)`: Corresponds directly to the `logits` argument of
  `torch.nn.BCEWithLogitsLoss()`.
* `weights`: A tensor which acts as coefficients for the loss. If a weight of x
  is provided for a datapoint and that datapoint is a true (false) positive
  (negative), it will be counted as x true (false) positives (negatives).
  Defaults to 1.0.
* `label_priors`: A tensor specifying the fraction of positive datapoints for
  each label. If not provided, it will be computed inside the loss function.


## Hyperparameters
While the functional form of the global objectives losses allow them to be
easily substituted in place of `torch.nn.BCEWithLogitsLoss()`, model
hyperparameters such as learning rate, weight decay, etc. may need to be
fine-tuned to the new loss. Fortunately, the amount of hyperparameter re-tuning
is usually minor.

The most important hyperparameters to modify are the learning rate and
dual_rate_factor (see the section on Loss Function Arguments, above).


## More Info
For more details, see the [Global Objectives paper](https://arxiv.org/abs/1608.04802).
