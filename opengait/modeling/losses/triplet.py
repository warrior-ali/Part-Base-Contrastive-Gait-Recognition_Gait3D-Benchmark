import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        len_embds = len(embeddings.shape)
        if len_embds == 3:
            embeddings = embeddings.permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        else:
            embeddings = embeddings.permute(
                1 , 0).contiguous().float()  # [n*s, 64] -> [64, n*s]


        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed ,len_embds)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        # this code is for escaping from 'cuda out of memory' caused by do operation on 12 milion parameters in out first idea. 

        # loss_num = torch.zeros((loss.size()[0])).cuda()
        # for i in range(loss.size()[0]):
        #   for j in range(loss.size()[1]):
        #     if loss[i,j] != 0:    
        #       loss_num[i] += 1
        # loss_num = loss_num.float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y, len_embds):
        """
            x: [p, n_x, c] or [p , n_x]
            y: [p, n_y, c] or [p , n_y]
        """
        if len_embds == 3:
            x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
            y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
            inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
            dist = x2 + y2 - 2 * inner
            dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        else: 
            x2 = x.unsqueeze(-1)
            y2 = y.unsqueeze(1)
            inner = x2.matmul(y.unsqueeze(-1).transpose(1, 2))
            dist = x2 + y2 - 2 * inner
            dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        
        p, n, _ = dist.size()
        n_r = torch.tensor(row_labels.size(), dtype=torch.int64)
        n_c = torch.tensor(clo_label.size(), dtype=torch.int64)

        # this code is for the case that we have to make +&- pairs for all frames.
        if n != n_r:
            temp_matches = torch.ones((n, n), dtype=torch.bool)
            for i in range(n_r):
                for j in range(n_c):
                  sub_matrix_range = int(n/n_r)
                  start_point_i = i * sub_matrix_range
                  start_point_j = j * sub_matrix_range
                  temp_matches[start_point_i :start_point_i + sub_matrix_range 
                  , start_point_j :start_point_j + sub_matrix_range] = matches[i,j]
            matches = temp_matches
        diffenc = torch.logical_not(matches)  # [n_r, n_c]

        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
