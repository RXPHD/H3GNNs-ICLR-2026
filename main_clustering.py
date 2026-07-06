import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor


from data_preprocess.datasets.heterodata_loader import Hetero
from data_preprocess.models.loss_func import sce_loss
from functools import partial
from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score, adjusted_rand_score
import torch.optim.lr_scheduler as lr_scheduler
from get_data import load_data, load_dataset, load_arxiv
from args_clustering import get_args
from encoders import Encoder
from premodels import LogisticReg
from sklearn.cluster import KMeans
from munkres import Munkres
from collections import Counter



class Model(nn.Module):
    def __init__(self, num_features, num_edges, args):
        super().__init__()
        # Student encoder
        self.student_encoder = Encoder(num_features, num_edges, args.latent_dim, args.num_hidden_homo, args.num_layers_homo, args.num_hidden_homo2, 
                                       args.num_layers_homo2, args.num_hidden_hetero, args.num_layers_hetero, args.dropout, args.dropout_fusion, 
                                       args.branch_heads, args.hidden_heads, args.out_heads, args.beta, args.homo_encoder_type, 
                                       args.hetero_encoder_type, args.share, args.sparse, args.norm, args.no_root, args.droppath).to(args.device)

        # Teacher encoder (fixed update by EMA)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)

        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

        # Momentum param for EMA
        self.momentum = args.momentum

        self._replace_rate = args.replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, num_features))
        
        self.device = args.device

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher params with student's, using EMA."""
        for t_param, s_param in zip(self.teacher_encoder.parameters(),
                                    self.student_encoder.parameters()):
            t_param.data = t_param.data * self.momentum + \
                           s_param.data * (1 - self.momentum)

    def forward(self, x_unmask, x_masked, adj_t):

        # Student forward
        z_student, student_weight = self.student_encoder(x_masked, adj_t, self.device)  # [N, d_model]

        # Teacher forward (no grad)
        with torch.no_grad():
            z_teacher, teacher_weight = self.teacher_encoder(x_unmask, adj_t, self.device)  # [N, d_model]

        return z_student, z_teacher, teacher_weight

    def generate_mixed_mask(self, diff, total_ratio, diff_ratio):

        N = diff.size(0)

        diff_num = int(N * total_ratio * diff_ratio)
        _, diff_idx = diff.topk(diff_num, largest=True) 
        mask_final = torch.zeros(N, dtype=torch.bool, device=diff.device)
        mask_final[diff_idx] = True

        base_num = int(N * total_ratio * (1 - diff_ratio))
        random_idx = (~mask_final).nonzero(as_tuple=True)[0]
        mask_random = random_idx[torch.randperm(random_idx.numel())[:base_num]]
        mask_final[mask_random] = True


        return mask_final  # shape [N], bool

    def generate_prob_mask(self, diff, total_ratio, diff_ratio):

        device = diff.device
        N = diff.size(0)
        target_mask_count = int(total_ratio * N)

        # 1) baseline 
        #    p_base = total_ratio * (1 - diff_ratio).
        p_base = total_ratio * (1.0 - diff_ratio)

        # 2) diff-based 
        diff_max = diff.max().item()
        if diff_max < 1e-9:
            p_diff = torch.zeros(N, device=device)
        else:
            # delta_i = (diff_i / diff_max) * diff_ratio * total_ratio
            # => diff_i = diff_max => delta_i = diff_ratio*total_ratio
            #    diff_i = 0 => delta_i = 0
            p_diff = (diff / diff_max) * (diff_ratio * total_ratio)

        # 3) p_i = p_base + p_diff, clamp
        p = p_base + p_diff
        p = torch.clamp(p, 0.0, 1.0) 

        mask_bern = torch.rand(N, device=device)
        mask = (mask_bern < p)  # True or False

        return mask  # [N] bool


    def encoding_mask_noise(self, x, mask_rate, diff, attn_weights, epoch):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        if epoch < args.mask_warmup:
            # random masking
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes: ]

        else:
            if args.masking == "prob":
                mask_nodes = self.generate_prob_mask(diff, mask_rate, args.diff_ratio)
            elif args.masking == "mix":
                mask_nodes = self.generate_mixed_mask(diff, mask_rate, args.diff_ratio)

            mask_nodes = mask_nodes.nonzero(as_tuple=True)[0]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes


    def get_loss_fn(self, loss_type, alpha_l):
        if loss_type == "mse":
            criterion = nn.MSELoss()
        elif loss_type == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_type == "byol":
            def criterion(p, z):
                p = F.normalize(p, dim=-1)
                z = F.normalize(z, dim=-1)
                return 2 - 2 * (p * z).sum(dim=-1).mean()
        else:
            raise NotImplementedError(f"Loss function '{loss_type}' is not implemented.")
        return criterion



class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.diff = None
        self.teacher_weight = None

    def train_step(self, x, edge_index, epoch):

        self.model.train()
        self.optimizer.zero_grad()

        # Create masked input for student
        x_masked, mask_nodes = self.model.encoding_mask_noise(x, args.mask_rate, self.diff, self.teacher_weight, epoch)


        student_pred, teacher_pred, teacher_weight = self.model(x, x_masked.to(self.device), edge_index.to(self.device))     

        self.teacher_weight = teacher_weight

        # Compute BYOL loss
        loss = self.criterion(student_pred, teacher_pred)

        loss.backward()
        self.optimizer.step()

        # Update teacher with momentum
        self.model.update_teacher()

        self.diff = (student_pred - teacher_pred).pow(2).sum(dim=-1)

        return loss.item()



def update_learning_rate(step, warmup, optimizer, scheduler):
    if step < warmup:
        # Warm-up
        lr = args.lr * float(step) / float(max(1, warmup))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()

### Partial implementation from the MUSE GitHub repository: https://anonymous.4open.science/r/MUSE-BD4B

def cluster_eval(y_true, y_pred):

    y_true = y_true.detach().cpu().numpy() if type(y_true) is torch.Tensor else y_true
    y_pred = y_pred.detach().cpu().numpy() if type(y_pred) is torch.Tensor else y_pred

    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    # print(f"INFO {l1}, {l2}")
    # print(f"INFO numclasses {numclass1}, {numclass2}")
    # fill out missing classes
    ind = 0
    c2 = Counter(y_pred)
    maxclass = sorted(c2.items(), key=lambda item: item[1], reverse=True)[0][0]
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                ind = y_pred.tolist().index(maxclass)
                y_pred[ind] = i

    l2 = list(set(y_pred))
    numclass2 = len(l2)
    # print(f"INFO filled numclasses {numclass1}, {numclass2}")

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        # print(f"INOF: {len(l2)}, {len(indexes)}, {i}")
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = accuracy_score(y_true, new_predict)
    f1_macro = f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def unsup_eval(y_true, y_pred, quiet=False):
    y_true = y_true.detach().cpu().numpy() if type(y_true) is torch.Tensor else y_true
    y_pred = y_pred.detach().cpu().numpy() if type(y_pred) is torch.Tensor else y_pred

    acc, f1 = cluster_eval(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    # if not quiet:
    #     print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #             ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def kmeans_test(X, y, n_clusters, repeat=10, quiet=True):
    y = y.detach().cpu().numpy() if type(y) is torch.Tensor else y
    X = X.detach().cpu().numpy() if type(X) is torch.Tensor else X

    mask_nan = np.isnan(X)
    mask_inf = np.isinf(X)
    X[mask_nan] = 1
    X[mask_inf] = 1

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    for _ in range(repeat):


        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)



        acc_score, nmi_score, ari_score, macro_f1 = unsup_eval(
            y_true=y, y_pred=y_pred, quiet=quiet)
        acc_list.append(acc_score)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
        f1_list.append(macro_f1)
    return np.mean(acc_list), np.std(acc_list), np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(
        ari_list), np.mean(f1_list), np.std(f1_list)
    
    
def test(model, data, adj_t, num_classes, device):

    model.eval()
    with torch.no_grad():
        h_fused, _ = model.student_encoder(data.x, adj_t, device)

        result = kmeans_test(h_fused, data.y, n_clusters=num_classes, repeat=1)

    return result


def main(args):                          
    data, num_features, num_classes = load_data(args.dataset_name)
    data = data.to(args.device)
        
    if args.norm:   
        input = copy.deepcopy(data)
        input = T.ToSparseTensor()(input)
        adj_t = input.adj_t
        del input
        if args.no_root:
            adj_t = adj_t.set_diag(0)
        else:
            adj_t = adj_t.set_diag()
        adj_t = gcn_norm(adj_t, add_self_loops=False)
    elif args.sparse:
        adj_t = data.adj_t
    else:
        adj_t = data.edge_index
        
    adj_t = adj_t.to(args.device)
    
    
    final_acc = []
    final_nmi = []
    final_ari = []

    for trial in range(args.runs):
        model = Model(num_features, data.num_edges, args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warmup))

        criterion = model.get_loss_fn(args.loss_func, alpha_l=args.alpha_l)
        
        trainer = Trainer(model, optimizer, criterion, args.device)

        x_original = data.x.clone()

        epoch_iterator = tqdm(range(1, args.epochs + 1), desc="Training Epochs", unit="epoch")
        print("Start Self-Supervised Training (feature + neighbor reconstruction)...")

        best_acc = 0
        best_nmi = 0
        best_ari = 0
        
        for epoch in epoch_iterator:
            
            loss = trainer.train_step(x_original, adj_t, epoch) 
            
            update_learning_rate(epoch, args.warmup, optimizer, scheduler)

            if (epoch < 500) or (epoch % 10 == 0):
                model.eval()
                
                result = test(model, data, adj_t, num_classes, args.device)
                
                acc = result[0]
                nmi = result[2]
                ari = result[4]

                
                if acc > best_acc:
                    best_acc = acc
                if nmi > best_nmi:
                    best_nmi = nmi
                if ari > best_ari:
                    best_ari = ari

            
        final_acc.append(best_acc)
        final_nmi.append(best_nmi)
        final_ari.append(best_ari)

if __name__ == "__main__":
    args = get_args()
    main(args)
