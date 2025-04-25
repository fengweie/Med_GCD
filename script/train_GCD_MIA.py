# -*- coding: utf-8 -*-
# +
import torch.optim as optim
from torchvision import models
import os
from loguru import logger
import random
from torchvision import transforms
from load_data.get_datasets import get_datasets, get_class_splits
from sklearn.cluster import KMeans
from copy import deepcopy
from loss import info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, info_nce_logits_with_pseudo_labels
import torch.backends.cudnn as cudnn
from cluster_and_log_utils import log_accs_from_preds
from general_utils import AverageMeter, init_experiment
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import torch.nn.functional as F
sys.path.append('../')


def upper_triangle(matrix, device='cuda'):
    upper = torch.triu(matrix, diagonal=0).to(device)
    #diagonal = torch.mm(matrix, torch.eye(matrix.shape[0]))
    diagonal_mask = torch.eye(matrix.shape[0]).to(device)
    #diagonal_mask = torch.eye(matrix.shape[0])

    return upper * (1.0 - diagonal_mask)


def regularizer(proto, device='cuda'):

    c_seen = proto.shape[0]
    proto_expand1 = proto.unsqueeze(0).to(device)
    proto_expand2 = proto.unsqueeze(1).to(device)

    proto_norm_mat = torch.sum(
        (proto_expand2 - proto_expand1)**2,
        dim=-1).to(device)

    proto_norm_upper = upper_triangle(proto_norm_mat).to(device)

    d_mean = (2.0 / (c_seen**2 - c_seen) *
              torch.sum(proto_norm_upper)).to(device)

    sim_mat = torch.matmul(proto, proto.T).to(device)

    sim_mat_upper = (upper_triangle(sim_mat)).to(device)
    m = torch.max(sim_mat_upper)

    m = torch.min(sim_mat_upper)

    residuals = (
        upper_triangle(
            (-proto_norm_upper + d_mean + sim_mat_upper))).to(device)

    rw = (1 / c_seen) * (torch.sum(residuals)).to(device)

    return(rw)


def get_prototype(student_proj, student_out):
    features = student_proj.detach().cpu().numpy()
    preds = student_out.argmax(1).cpu().numpy()

    total_class = np.unique(preds)

    total_prototype = []
    for class_i in total_class:
        total_prototype.append(np.mean(features[preds == class_i], axis=0))

    proto_u = torch.tensor(np.array(total_prototype)).cuda()

    return proto_u


def pairwise_cosine_sim(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.matmul(x, y.T)


def EuclideanDistances(a, b):

    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return sum_sq_a + sum_sq_b - 2 * a.mm(bt)



class CE_Head(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True):
        super().__init__()
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.last_layer(x)
        return logits


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr']
                      for param_group in optimizer.param_groups])).item()


def train_dual(student_ce, train_loader, test_loader, args):

    from cluster_and_log_utils import set_args_mmf
    set_args_mmf(args, train_loader)

    optimizer_ce = optim.SGD(
        student_ce.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    exp_lr_scheduler_ce = lr_scheduler.CosineAnnealingLR(
        optimizer_ce,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    args.current_epoch = 0
    best_test_acc_all_cl = -1
    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )
    for epoch in range(args.epochs):
        student_ce.train()
        for batch_idx, batch in enumerate(train_loader):
            images_, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(
                non_blocking=True), mask_lab.cuda(
                non_blocking=True).bool()
            images = torch.cat(images_, dim=0).cuda(non_blocking=True)

            student_proj, student_out, student_feat = student_ce(images)
            # regularizer
            proto = student_ce.ce_head.last_layer.weight
            prototype_all = torch.nn.functional.normalize(proto, dim=-1)

            labels_p = torch.arange(0, args.mlp_out_dim).cuda()
            labels_p = labels_p.contiguous().view(-1, 1)
            mask_p = (1 - torch.eq(labels_p, labels_p.T).float()).cuda()

            logits_p = torch.div(
                torch.matmul(prototype_all, prototype_all.T),
                1)
#
            mean_prob_neg = torch.log(
                (mask_p * torch.exp(logits_p)).sum(1) / mask_p.sum(1))
            mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
            loss_u = mean_prob_neg.mean()

            sim_mat = pairwise_cosine_sim(
                student_feat, proto) / args.temperature
            s_dist = F.softmax(sim_mat, dim=1)
            cost_mat = pairwise_cosine_sim(student_feat, proto)

            loss_u += (cost_mat * s_dist).sum(1).mean()

            teacher_out = student_out.detach()

            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (
                student_out / args.temperature).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab]
                                   for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup
            # compute logits
            student_proj_norm = torch.nn.functional.normalize(
                student_proj, dim=-1)
            similarity_matrix = torch.matmul(
                student_proj_norm, student_proj_norm.T)

            mask_i = ~torch.eye(
                student_proj_norm.size(0),
                dtype=torch.bool).to(device)

            b_ = 0.5 * int(student_proj_norm.size(0))
            logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
            logits = similarity_matrix - logits_max.detach()

            nominator = torch.exp(logits)
            denominator = torch.sum(torch.exp(logits), dim=1)
            p_weight = nominator / denominator
            teacher_out = 0.3 * \
                torch.matmul(p_weight.detach(), student_out) + 0.7 * student_out

            cluster_loss = cluster_criterion(
                student_out, teacher_out.detach(), epoch)
            avg_probs = (
                student_out /
                args.student_out_temp).softmax(
                dim=1).mean(
                dim=0)
            me_max_loss = - \
                torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits_with_pseudo_labels(
                features=student_proj, pseudo_labels=teacher_out)
            contrastive_loss = torch.nn.CrossEntropyLoss()(
                contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1)
                                     for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + \
                args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * contrastive_loss + \
                args.sup_weight * sup_con_loss
            loss += loss_u
            optimizer_ce.zero_grad()
            loss.backward()
            optimizer_ce.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info(
                    'Train Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'.format(
                        epoch, batch_idx, len(train_loader), loss.item(), pstr))

        if epoch % args.test_freq == 0:

            args.logger.info(
                'Testing on unlabelled examples in the training data...')

            with torch.no_grad():
                all_acc_test_cl, old_acc_test_cl, new_acc_test_cl = test(
                    student_ce,
                    test_loader,
                    epoch=epoch,
                    save_name='Test ACC',
                    args=args,
                    train_loader=train_loader)

            args.logger.info(
                'Test Accuracies CL: All {:.1f} | Old {:.1f} | New {:.1f}'.format(
                    all_acc_test_cl, old_acc_test_cl, new_acc_test_cl))

        # Step schedule
        exp_lr_scheduler_ce.step()

    torch.save(student_ce.state_dict(), args.model_dir + '/model_last.pt')

    args.logger.info(
        "model saved to {}.".format(
            args.model_dir +
            '/model_last.pt'))

    args.logger.info(
        f'Metrics with last epoch model on test set: All: {all_acc_test_cl:.1f} Old: {old_acc_test_cl:.1f} New: {new_acc_test_cl:.1f} ')


def test(model, test_loader, epoch, save_name, args, train_loader):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, batch in enumerate(test_loader):
        images = batch[0]
        label = batch[1]
        images = images.cuda()
        # transform (linear layer)
        _, logits, _ = model(images)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(label.cpu().numpy())

        mask = np.append(mask, np.array([True if x.item() in range(
            len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args, train_loader=train_loader)

    return all_acc, old_acc, new_acc


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits, x


class Resnet_model(nn.Module):
    def __init__(self, args):
        super().__init__()

        # backbone
        if args.backbone == "vit":
            self.backbone = torch.hub.load(
                './torch/facebookresearch-dino-7c446df',
                'dino_vitb16',
                trust_repo=True,
                source='local')
            # ----------------------
            # HOW MUCH OF BASE MODEL TO FINETUNE
            # ----------------------
            for m in self.backbone.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            for name, m in self.backbone.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= 11:
                        m.requires_grad = True
            self.ce_head = CE_Head(
                in_dim=768, out_dim=args.num_labeled_classes)
        else:
            if args.backbone == "resnet18":
                self.backbone = models.resnet18(pretrained=True)
            elif args.backbone == "resnet34":
                self.backbone = models.resnet34(pretrained=True)
            elif args.backbone == "resnet50":
                self.backbone = models.resnet50(pretrained=True)
            elif args.backbone == "resnet101":
                self.backbone = models.resnet101(pretrained=True)
            elif args.backbone == "resnet152":
                self.backbone = models.resnet152(pretrained=True)
            else:
                raise NotImplementedError
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            self.ce_head = DINOHead(
                in_dim=in_features,
                out_dim=args.mlp_out_dim,
                nlayers=args.num_mlp_layers)

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        ce_backbone_feature = self.backbone(images)
        x_proj, logits, x = self.ce_head(ce_backbone_feature)

        return x_proj, logits, x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--img_size', default=224, type=int)

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', type=list, default=['v2'])
    parser.add_argument('--dataset_name', type=str, default='pathmnist')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model_dir', type=str, default='./checkpoints')

    parser.add_argument(
        '--student_out_temp',
        default=0.1,
        type=float,
        help='Final value (after linear warmup)of the student out temperature.')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument(
        '--warmup_teacher_temp',
        default=0.07,
        type=float,
        help='Initial value for the teacher temperature.')
    parser.add_argument(
        '--teacher_temp',
        default=0.04,
        type=float,
        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument(
        '--warmup_teacher_temp_epochs',
        default=30,
        type=int,
        help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--p', default=1.1, type=float)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--est_freq', default=10, type=int)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--tro', default=0.5, type=float)
    parser.add_argument('--stop_epoch', default=500, type=int)
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
        help="softmax temperature")
    parser.add_argument('--backbone', type=str, default='resnet18')
    # ----------------------
    # INIT
    # ----------------------

    args = parser.parse_args()
    pid = os.getpid()
    print('MY PID：', pid)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ])

    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=args.n_views)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, args, datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args)
    total_class = args.total_classes

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    args.model_dir = os.path.join(args.model_dir, 'mia_img', args.dataset_name)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    log_dir = os.path.join('./log', 'mia_img', args.dataset_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    print(f'Experiment saved to: {args.log_dir}')

    args.logger.info(
        f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # TRAIN
    # ----------------------
    student_ce = Resnet_model(args=args)
    student_ce = student_ce.to(device)
    args.logger.info('model build')

    seed = torch.randint(0, 100000, (1,)).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    # --------------------
    # SAMPLER
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)

    sample_weights = [
        1 if i < label_len else label_len /
        unlabelled_len for i in range(
            len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=True)

    test_loader_unlabelled = DataLoader(
        test_dataset,
        num_workers=args.num_workers,
        batch_size=256,
        shuffle=False,
        pin_memory=False)
    print("len of train_set:{},len of test_set:{}".format(
        len(train_dataset), len(test_dataset)))

    train_dual(student_ce, train_loader, test_loader_unlabelled, args)
