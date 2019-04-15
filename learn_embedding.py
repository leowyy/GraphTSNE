import os
import time
import torch
from math import ceil

from core.tsne_torch_loss import compute_joint_probabilities, tsne_torch_loss
from util.network_utils import get_net_projection, save_checkpoint
from util.evaluation_metrics import evaluate_viz_metrics
from util.training_utils import get_torch_dtype


dtypeFloat, dtypeLong = get_torch_dtype()


def train(net, train_set, opt_parameters, checkpoint_dir, val_set=None):
    # Optimization parameters
    n_batches = opt_parameters['n_batches']
    shuffle_flag = opt_parameters['shuffle_flag']
    sampling_flag = opt_parameters['sampling_flag']
    metric = opt_parameters['distance_metric']
    graph_weight = opt_parameters['graph_weight']
    perplexity = opt_parameters['perplexity']
    val_batches = opt_parameters['val_batches']

    lr = opt_parameters['learning_rate']
    max_iters = opt_parameters['max_iters']
    batch_iters = opt_parameters['batch_iters']
    decay_rate = opt_parameters['decay_rate']
    start_epoch = opt_parameters['start_epoch']
    checkpoint_interval = ceil(max_iters / 1)

    # Optimizer
    optimizer = net.update(lr)

    # Statistics
    t_start = time.time()
    t_start_total = time.time()
    average_loss_old = 1e10
    running_tsne_loss = 0.0
    running_graph_loss = 0.0
    running_loss = 0.0
    running_total = 0
    tab_results = []

    all_features_P_initialised = False
    print("Start training...")

    for iteration in range(start_epoch+1, start_epoch+max_iters+1):
        net.train()  # Set the net to training mode

        # Create a new set of data blocks per iteration
        if shuffle_flag or not all_features_P_initialised:
            train_set.create_all_data(n_batches=n_batches, shuffle=shuffle_flag, sampling=sampling_flag)
            all_features_P = []
            all_graph_P = []

            for G in train_set.all_data:
                X = G.inputs.view(G.inputs.shape[0], -1).numpy()
                if graph_weight != 1.0:
                    P = compute_joint_probabilities(X, perplexity=perplexity, metric=metric, method='approx', adj=G.adj_matrix)
                    all_features_P.append(P)

                if graph_weight != 0.0:
                    P = compute_joint_probabilities(X, perplexity=perplexity, metric='shortest_path', method='exact', adj=G.adj_matrix, verbose=0)
                    all_graph_P.append(P)

            all_features_P_initialised = True

        # Forward pass through all training data
        for i, G in enumerate(train_set.all_data):

            y_pred = net.forward(G)

            feature_loss = torch.tensor([0]).type(dtypeFloat)
            graph_loss = torch.tensor([0]).type(dtypeFloat)

            if graph_weight != 1.0:
                feature_loss = tsne_torch_loss(all_features_P[i], y_pred)
            if graph_weight != 0.0:
                graph_loss = tsne_torch_loss(all_graph_P[i], y_pred)

            loss = (1 - graph_weight) * feature_loss + graph_weight * graph_loss
            running_tsne_loss += feature_loss.item()
            running_graph_loss += graph_loss.item()

            running_loss += loss.item()
            running_total += 1

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # update learning rate, print results, perform validation
        if not iteration % batch_iters:

            # time
            t_stop = time.time() - t_start
            t_start = time.time()

            # update learning rate
            average_loss = running_loss / running_total
            if average_loss > 0.99 * average_loss_old:
                lr /= decay_rate
            average_loss_old = average_loss
            optimizer = net.update_learning_rate(optimizer, lr)

            # print results
            average_tsne_loss = running_tsne_loss / running_total
            average_graph_loss = running_graph_loss / running_total
            running_tsne_loss = 0.0
            running_graph_loss = 0.0
            print('iteration= %d, loss(%diter)= %.8f, tsne_loss= %.8f, graph_loss= %.8f, \n'
                  'lr= %.8f, time(%diter)= %.2f' %
                  (iteration, batch_iters, average_loss, average_tsne_loss, average_graph_loss, lr, batch_iters, t_stop))
            tab_results.append([iteration, average_loss, time.time() - t_start_total])

            running_loss = 0.0
            running_total = 0

            if val_set is not None:
                validate(net, val_set, val_batches)

        # save checkpoint
        if iteration % checkpoint_interval == 0:
            print('Saving checkpoint at iteration = {}\n'.format(iteration))
            filename = os.path.join(checkpoint_dir, net.name + '_' + str(iteration) + '.pkl')
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)

    return tab_results


def validate(net, val_set, val_batches):
    y_emb = get_net_projection(net, val_set, n_batches=val_batches)
    _ = evaluate_viz_metrics(y_emb, val_set)
    print()
