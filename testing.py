import copy
from tqdm import tqdm
import torch
import torch.nn as nn

from data_preprocess.utils import create_optimizer, accuracy


def node_classification_evaluation(embedding, num_classes, labels, train_mask, val_mask, test_mask, lr, weight_decay, device):
    
    with torch.no_grad():
        # x = model.embed(x.to(device), graph.edge_index.to(device))
        in_feat = embedding.shape[1]
        print(in_feat)
    classifier = LogisticRegression(in_feat, num_classes)
    
    classifier.to(device)
    optimizer_f = create_optimizer("adam", classifier, lr, weight_decay)
    final_acc, estp_acc = linear_probing_for_transductive_node_classification(classifier, optimizer_f, embedding, labels, train_mask, val_mask, test_mask, 
                                                                              device)
    return final_acc, estp_acc


def linear_probing_for_transductive_node_classification(classifier, optimizer_f, embedding, labels, train_mask, val_mask, test_mask, device):
    criterion = torch.nn.CrossEntropyLoss()

    x = embedding.to(device)


    best_val_acc = 0
    best_val_epoch = 0
    best_model = None
    max_epoch = 500

    for epoch in range(max_epoch):
        classifier.train()
        out = classifier(embedding)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer_f.zero_grad()
        loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer_f.step()

        with torch.no_grad():
            classifier.eval()
            pred = classifier(embedding)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(classifier)


    best_model.eval()
    with torch.no_grad():
        pred = best_model(x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self,x):
        logits = self.linear(x)
        return logits
