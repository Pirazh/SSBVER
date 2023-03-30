import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class CMPTLoss(nn.Module):
    def __init__(self,ncrops):
        super().__init__()
        self.ncrops = ncrops
    
    def forward(self, student_output, teacher_output):
        student_out = student_output.chunk(self.ncrops)
        teacher_out = teacher_output.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # skip cases where student and teacher operate on the same view
                    continue
                loss = (q - student_out[v]).norm(dim=1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        return total_loss

class SSLLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim).cuda())
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        self.center = self.center * self.center_momentum + \
                            batch_center * (1 - self.center_momentum)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, 
                    targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(log_probs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1).mean(dim=0)
        return loss

def euclidean_dist(x, y):
        m, n = x.shape[0], y.shape[0]
        distance = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distance = torch.addmm(distance, x, y.t(), 
            beta=1, alpha=-2).clamp(min=1e-12).sqrt()
        return distance

def hard_example_mining(distance, labels):
        assert len(distance.size()) == 2
        assert distance.size(0) == distance.size(1)
        N = distance.size(0)

        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        dist_ap, _ = torch.max(
            distance[is_pos].contiguous().view(N, -1), 1, keepdim=True)

        dist_an, _ = torch.min(
            distance[is_neg].contiguous().view(N, -1), 1, keepdim=True)
        
        return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss(object):
    def __init__(self, use_margin=False, margin=None):
        self.use_margin = use_margin
        self.margin = margin
        if self.use_margin:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.use_margin:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss
