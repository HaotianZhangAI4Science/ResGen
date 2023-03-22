import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)

def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)

def get_model_loss(model, batch, config):
    compose_noise = torch.randn_like(batch.compose_pos) * config.train.pos_noise_std
    loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = model.get_loss(
            pos_real = batch.pos_real,
            y_real = batch.cls_real.long(),
            # p_real = batch.ind_real.float(),    # Binary indicators: float
            pos_fake = batch.pos_fake,

            edge_index_real = torch.stack([batch.real_compose_edge_index_0, batch.real_compose_edge_index_1], dim=0),
            edge_label = batch.real_compose_edge_type,
            
            index_real_cps_edge_for_atten = batch.index_real_cps_edge_for_atten,
            tri_edge_index = batch.tri_edge_index,
            tri_edge_feat = batch.tri_edge_feat,

            compose_vec= batch.compose_vec,
            compose_feature = batch.compose_feature.float(),
            compose_pos = batch.compose_pos + compose_noise,
            idx_ligand = batch.idx_ligand_ctx_in_compose,
            idx_protein = batch.idx_protein_in_compose,

            y_frontier = batch.ligand_frontier,
            idx_focal = batch.idx_focal_in_compose,
            pos_generate=batch.pos_generate,
            idx_protein_all_mask = batch.idx_protein_all_mask,
            y_protein_frontier = batch.y_protein_frontier,

            compose_knn_edge_index = batch.compose_knn_edge_index,
            compose_knn_edge_feature = batch.compose_knn_edge_feature,
            real_compose_knn_edge_index = torch.stack([batch.real_compose_knn_edge_index_0, batch.real_compose_knn_edge_index_1], dim=0),
            fake_compose_knn_edge_index = torch.stack([batch.fake_compose_knn_edge_index_0, batch.fake_compose_knn_edge_index_1], dim=0),
        )
    return loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf
