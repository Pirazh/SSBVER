import math
import sys

import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn.functional as F


def calc_stats(args, teacher_output, student_output, center):
    center_out = F.softmax(center + 1e-18, dim=-1)
    center_entropy = (- center_out * torch.log(center_out)).sum(dim=-1)
    
    teach_out = F.softmax(teacher_output + 1e-18, dim=-1)
    teach_entropy = (- teach_out * torch.log(teach_out)).sum(dim=-1).mean()

    student_out = F.softmax(student_output + 1e-18, dim=-1)
    student_entropy = (- student_out * torch.log(student_out)).sum(dim=-1).mean()

    teach_out = teach_out.chunk(2)
    student_out = student_out.chunk(args.local_crops_num + 2)
    
    teach_student_KL = 0
    n_terms = 0
    for iq, q in enumerate(teach_out):
        for v in range(len(student_out)):
            if v != iq:
                teach_student_KL += (q * 
                            torch.log(q / student_out[v])).sum(dim=-1).mean()
                n_terms += 1
    teach_student_KL /= n_terms
    return teach_entropy, student_entropy, teach_student_KL, center_entropy

def clip_gradients(model, clip):
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def train_one_epoch(args,
            epoch,
            train_loader,
            student,
            teacher,
            loss_fn,
            optimizer,
            lr_scheduler,
            logger, 
            writer=None):
    
    student.train()
    teacher.train()
    
    epoch_loss = 0

    for i, batch in enumerate(train_loader):
        iteration = epoch * len(train_loader) + i
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[epoch]
        
        optimizer.zero_grad()

        images, vids, _ = batch
        images = [im.cuda(non_blocking=True) for im in images]
        vids = vids.cuda()

        teacher_ssl_output = teacher(images[:2])
        student_ssl_output, student_reid_output, vids1 = student(images, vids)
        cls_scores, feats = student_reid_output
        
        id_loss, triplet_loss, ssl_loss, cmpt_loss, ssl_loss_class = loss_fn(
                                                    cls_scores, 
                                                    feats, 
                                                    vids1,
                                                    student_ssl_output,
                                                    teacher_ssl_output,
                                                    epoch)
        loss = id_loss + triplet_loss

        writer.add_scalar('Loss/CLS_Loss', 
                    id_loss.item()/args.id_loss_lambda, iteration)
        writer.add_scalar('Loss/TRIPLET_Loss', 
                    triplet_loss.item()/args.triplet_loss_lambda, iteration)

        if args.ssl_loss_lambda > 0 :
            loss += ssl_loss
            writer.add_scalar('Loss/SSL_Loss', 
                    ssl_loss.item()/(args.ssl_loss_lambda + 1e-12), iteration)

            teach_entropy, student_entropy, teach_student_KL, center_entropy = \
                calc_stats(args, teacher_ssl_output.detach().clone(), 
                                student_ssl_output.detach().clone(),
                                ssl_loss_class.center)

            writer.add_scalar('Stats/Teacher_Entropy', 
                                    teach_entropy.item(), iteration)
            writer.add_scalar('Stats/Student_Entropy', 
                                    student_entropy.item(), iteration)
            writer.add_scalar('Stats/Teacher_Student_KL', 
                                    teach_student_KL.item(), iteration)
            writer.add_scalar('Stats/Center_Entropy', 
                                    center_entropy.item(), iteration)

        if args.cmpt_loss_lambda > 0:
            loss += cmpt_loss
            writer.add_scalar('Loss/CMPT_Loss', cmpt_loss.item(), iteration)

        epoch_loss += loss.item()
        loss.backward()
        acc = (cls_scores.max(1)[1] == vids1).float().mean()

        if args.clip_grad > 0:
            clip_gradients(student, args.clip_grad)

        optimizer.step()

        with torch.no_grad():
            m = args.momentum_teacher
            for k, v in teacher.state_dict().items():
                v.copy_(v * m + (1 - m) * \
                    student.state_dict()['{}'.format(k)].detach())
            
        if (i + 1) % args.log_freq == 0:
            message = 'epoch: [{0:3d}/{1}] '.format(epoch + 1, args.epochs)
            message += 'iteration: [{0:3d}/{1}] '.format(
                                i + 1, len(train_loader))
            message += 'lr: {0:.2e} '.format(lr_scheduler[epoch])
            if args.ssl_loss_lambda > 0:
                message += 'ssl_loss: {0:.4f} '.format(
                                ssl_loss.item()/(args.ssl_loss_lambda + 1e-12))
            if args.cmpt_loss_lambda > 0:
                message += 'compactness_loss: {0:.4f} '.format(
                            cmpt_loss.item()/(args.cmpt_loss_lambda + 1e-12))
            message += 'id_loss: {0:.4f} '.format(
                            id_loss.item()/args.id_loss_lambda)
            message += 'triplet_loss: {0:.4f} '.format(
                            triplet_loss.item()/args.triplet_loss_lambda) 
            message += 'Accuracy: {0:.4f}'.format(acc)
            logger.info(message)
    
    epoch_loss /= len(train_loader)
    logger.info('Average loss: {0:.4f}'.format(epoch_loss))


def eval_reid(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        if q_camid != -1:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
        else:
            remove = np.asarray([False] * num_g)
            keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        
        # ------------- Added for Nvidia -----------------
        # orig_cmc = orig_cmc[:100]
        # ------------------------------------------------

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if not num_rel == 0:
            AP = tmp_cmc.sum() / num_rel
        else:
            AP = 0.0
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities not in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def re_ranking(probFea, galFea, k1, k2, lambda_value, 
                local_distmat=None, only_local=False):

    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea]).cpu()
        distmat = torch.pow(feat,2).sum(dim=1,
                                    keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, 
                                    keepdim=True).expand(all_num, all_num).t()
        distmat = torch.addmm(distmat, feat, feat.t(), beta=1, alpha=-2)
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = \
                initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = \
                initial_rank[\
                    candidate_forward_k_neigh_index,:int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = \
                candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(
                candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * \
                    len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + \
                np.minimum(V[i, indNonZero[j]],V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist