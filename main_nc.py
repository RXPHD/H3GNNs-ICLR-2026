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
from sklearn.metrics import accuracy_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
from get_data import load_data, load_dataset, load_arxiv
from args_nc import get_args
from encoders import Encoder
from premodels import LogisticReg



class Model(nn.Module):
    def __init__(self, num_features, num_edges, args):
        super().__init__()
        # Student encoder
        self.student_encoder = Encoder(num_features, num_edges, args.latent_dim, args.num_hidden_homo, args.num_layers_homo, args.num_hidden_homo2, 
                                       args.num_layers_homo2, args.num_hidden_hetero, args.num_layers_hetero, args.dropout, args.dropout_fusion, 
                                       args.branch_heads, args.hidden_heads, args.out_heads, args.beta, args.homo_encoder_type, 
                                       args.hetero_encoder_type, args.share, args.sparse, args.norm, args.no_root).to(args.device)

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
            elif args.masking == "attn":
                mask_nodes = self.generate_attn_mask(attn_weights, mask_rate, args.diff_ratio)
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



def test(model, data, adj_t, train_mask, val_mask, test_mask, num_classes):

    model.eval()
    with torch.no_grad():
        h_fused, _ = model.student_encoder(data.x, adj_t, args.device)  # [N, in_channels]

    classifier = LogisticReg(in_dim=args.latent_dim, out_dim=num_classes).to(args.device)
    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay_classifier)

    y = data.y

    best_val_micro = 0.0
    best_test_micro = 0.0

    for epoch in range(args.epochs_LR):
        classifier.train()
        optimizer_classifier.zero_grad()
        out = classifier(h_fused)  # [N, num_classes]

        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer_classifier.step()

        classifier.eval()
        with torch.no_grad():
            logits = classifier(h_fused)
            pred = logits.argmax(dim=1)

            train_micro = f1_score(y[train_mask].detach().cpu().numpy(), pred[train_mask].detach().cpu().numpy(), average='micro')
            val_micro  = f1_score(y[val_mask].detach().cpu().numpy(), pred[val_mask].detach().cpu().numpy(), average='micro')
            test_micro  = f1_score(y[test_mask].detach().cpu().numpy(), pred[test_mask].detach().cpu().numpy(), average='micro')

            if val_micro > best_val_micro:
                best_val_micro = val_micro
                best_test_micro = test_micro

    return best_val_micro, best_test_micro, h_fused


def main(args):
    if args.dataset_name in ['chameleon-filtered', 'squirrel-filtered']:
        data, num_features, num_classes = load_dataset(args.dataset_name, args.device)
        data.num_edges = data.edge_index.size(1)
        data.to_device()
        mask_dim = 0                                          # mask shape: [num_splits, N]
    elif args.dataset_name == "ogbn-arxiv":
        data, num_features, num_classes = load_arxiv()
        data = data.to(args.device)
        mask_dim = 1                                         
    else:
        data, num_features, num_classes = load_data(args.dataset_name)
        data = data.to(args.device)
        mask_dim = 1   
        
    if args.norm:
        if args.dataset_name in ['chameleon-filtered', 'squirrel-filtered']:
            edge_index = data.edge_index
            num_nodes = data.x.size(0)
            adj_t = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                sparse_sizes=(num_nodes, num_nodes)
            )
        elif args.sparse:
            adj_t = data.adj_t
        else:    
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
    
    if data.train_mask.dim() == 1:
        num_trials = 1                                        
    elif mask_dim == 0:
        num_trials = data.train_mask.shape[0]                 # [num_splits, N]
    else:
        num_trials = data.train_mask.shape[1]                 # [N, num_splits]
    
    final_test = []
    test_acc_matrix = np.zeros((num_trials, args.epochs))

    for trial in range(num_trials):
        model = Model(num_features, data.num_edges, args).to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warmup))

        criterion = model.get_loss_fn(args.loss_func, alpha_l=args.alpha_l)

        trainer = Trainer(model, optimizer, criterion, args.device)

        x_original = data.x.clone()

        epoch_iterator = tqdm(range(1, args.epochs + 1), desc="Training Epochs", unit="epoch")
        print("Start Self-Supervised Training (feature + neighbor reconstruction)...")

        best_acc_test = 0

        for epoch in epoch_iterator:

            loss = trainer.train_step(x_original, adj_t, epoch) 

            update_learning_rate(epoch, args.warmup, optimizer, scheduler)

            if (epoch < 300) or (epoch % 5 == 0):
                model.eval()
                cur_split = 0 if (num_trials == 1) else (trial % num_trials)
                
                if data.train_mask.dim() == 1:
                    train_mask = data.train_mask
                    val_mask = data.val_mask
                    test_mask = data.test_mask
                elif mask_dim == 0:
                    train_mask = data.train_mask[cur_split]
                    val_mask = data.val_mask[cur_split]
                    test_mask = data.test_mask[cur_split]
                else:
                    train_mask = data.train_mask[:, cur_split]
                    val_mask = data.val_mask[:, cur_split]
                    test_mask = data.test_mask[:, cur_split]

                acc_val, acc_test, h_fused = test(model, data, adj_t, train_mask, val_mask, test_mask, num_classes)


                test_acc_matrix[trial, epoch-1] = acc_test

                if acc_test > best_acc_test:
                    best_acc_test = acc_test


        final_test.append(best_acc_test)

        # print('\n[FINAL RESULT] Dataset:{} | Run:{} | ACC:{:.2f}+-{:.2f}'.format(args.dataset, args.ntrials, np.mean(results),
        #                                                                            np.std(results)))
    print(final_test)
    print(np.mean(final_test))
    print(np.std(final_test))

if __name__ == "__main__":
    args = get_args()
    main(args)
