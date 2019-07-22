import torch
import torch.optim as optim
import numpy as np


def train_model(data, use_global_objectives,
                metric_func, at_target_rate,
                obj_type, at_target_type,
                train_iteration, lr, num_checkpoints,
                criterion=None):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    w = torch.tensor([-1.0, -1.0],  requires_grad=True, device=device)
    b = torch.tensor([0.0], requires_grad=True, device=device)
    x = torch.tensor(data['train_data'], requires_grad=False, dtype=torch.float, device=device)
    labels = torch.tensor(data['train_labels'], requires_grad=False, dtype=torch.float, device=device)

    params = [w, b]
    if not use_global_objectives:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = criterion.to(device)
        params += list(criterion.parameters())

    optimizer = optim.SGD(params, lr=lr)
    checkpoint_step = train_iteration // num_checkpoints

    for t in range(train_iteration):
        # Zero gradient at the start of the iteration
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        logits = x.mv(w) + b
        # Compute and print loss
        loss = criterion(logits, labels)
        # perform a backward pass, and update the weights.
        loss.backward()
        optimizer.step()

        if t % checkpoint_step == 0:
            w_numpy = w.cpu().detach().numpy()
            b_numpy = b.cpu().detach().numpy()
            obj_rate, target_rate, _ = metric_func(
                np.dot(data['train_data'], w_numpy) + b_numpy,
                data['train_labels'],
                at_target_rate
            )

            print('Loss = {}, {} = {:.3f} @ {} = {:.3f}'.format(
                loss.data, obj_type, obj_rate,
                at_target_type, target_rate))

    w_numpy = w.cpu().detach().numpy()
    b_numpy = b.cpu().detach().numpy()

    obj_rate, target_rate, threshold = metric_func(
        np.dot(data['eval_data'], w_numpy) + b_numpy,
        data['eval_labels'],
        at_target_rate
    )

    return obj_rate, target_rate, w_numpy, b_numpy, threshold
