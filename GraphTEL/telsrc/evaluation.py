
# In order to ensure the fairness and comparability of the test results, 
# we directly copied and followed the evaluation methods
# from other recent works with good impact including GraphMAE.

# At the same time, in order not to cause unnecessary misunderstandings 
# during the blind review process and to respect the work of others, 
# we append URLs that link to their code.


# Directly copy from https://github.com/THUDM/GraphMAE/blob/main/graphmae/evaluation.py
def graph_classification_evaluation(model, Pooling, eval_loader, num_classes, lr, weight_decay, epoch, device, mute=False):

    model.eval()
    x_list = []
    y_list = []
    for i, (batch_g, labels) in enumerate(eval_loader):
        batch_g = batch_g.to(device)
        _, graphemb = model.learn_rep(batch_g, batch_g.ndata["attr"])

        y_list.append(labels.numpy())
        x_list.append(graphemb.cpu().numpy())

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_acc, test_std = evaluate_graph_embeddings_using_svm(x, y)

    return test_acc, test_std


# Directly copy from https://github.com/susheels/adgcl/tree/main/unsupervised
def graph_regression_evaluation(model, graph):

    model.eval()
    graphemb, _ = model.learn_rep(graph, graph.ndata["attr"])

    # You can comment out all the parts about the encoder, because the embedding has been calculated previously.
    _, _, test_result = kf_embedding_evaluation(encoder=None, dataset=graphemb, folds=10, batch_size=128)

    return test_result

