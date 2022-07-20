import torch
import torch.nn as nn
import IPython
from torch.autograd import Variable
import torchac
from utils import torchac_utils
from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
import time


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adv_loss = torch.nn.BCELoss()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)

    def l2_residual_loss(self, range_image_pred, range_image_gt, nonground_mask=None, loss_min_clip=0.1):
        loss = range_image_pred - range_image_gt
        if nonground_mask is not None:
            loss = loss * nonground_mask
            # test min clip, if work TODO
            loss = loss.square()
            # loss = torch.where(loss > loss_min_clip, loss, 0.1 * loss)
            # loss = torch.where(loss > loss_min_clip, loss, torch.tensor(0., dtype=torch.float, device=loss.device))
            loss = loss.sum() / nonground_mask.sum()
        else:
            # test min clip, if work TODO
            # loss = torch.where(loss > loss_min_clip, loss, torch.tensor(0., dtype=torch.float, device=loss.device))
            loss = loss.square().mean()
        return loss

    def l1_residual_loss(self, range_image_pred, range_image_gt, nonground_mask=None, loss_min_clip=0.1):
        loss = range_image_pred - range_image_gt
        if nonground_mask is not None:
            loss = loss * nonground_mask
            # test min clip, if work TODO
            # loss = torch.where(loss > loss_min_clip, loss, 0.1 * loss)
            # loss = torch.where(loss > loss_min_clip, loss, torch.tensor(0., dtype=torch.float, device=loss.device))
            loss = loss.abs().sum() / nonground_mask.sum()
        else:
            # test min clip, if work TODO
            # loss = torch.where(loss > loss_min_clip, loss, torch.tensor(0., dtype=torch.float, device=loss.device))
            loss = loss.abs().mean()
        return loss

    def chamfer_distance_loss(self, pc_1, pc_2, mask=None):
        chamLoss = dist_chamfer_3D.chamfer_3DDist()
        if mask is None:
            dist1, dist2, idx1, idx2 = chamLoss(pc_1, pc_2)
            loss = torch.mean((dist1 + dist2) / 2)
        else:
            dist1, dist2, idx1, idx2 = chamLoss(pc_1 * mask, pc_2 * mask)
            loss = torch.sum((dist1 + dist2) / 2) / mask.sum()
        return loss

    def entropy_loss(self, range_image_pred, range_image_gt, nonground_mask=None):
        residual = range_image_pred - range_image_gt
        loss = torch.std(residual)
        # if nonground_mask is not None:
        #     residual = residual * nonground_mask
        #     loss = torch.std(residual)
        # else:
        #     loss = torch.std(residual)
        return loss

    def adversarial_loss(self, pred, valid=True):
        if valid:
            gt = Variable(torch.cuda.FloatTensor(pred.shape[0], 1).fill_(1.0), requires_grad=False)
        else:
            gt = Variable(torch.cuda.FloatTensor(pred.shape[0], 1).fill_(0.0), requires_grad=False)
        
        return self.adv_loss(pred, gt)
        # if valid:
        #     # Real adversarial
        #     loss = torch.mean((pred - 1) ** 2)
        #     return loss
        # else:
        #     # Fake adversarial
        #     loss = torch.mean((pred) ** 2)
        #     return loss
    
    

    def classification_loss(self, pred, gt):
        loss = self.cls_loss(pred.view(-1, pred.shape[-1]), gt.view(-1).to(torch.long))
        return loss

    def bitrate_loss(self, prob, sym):
        estimated_bits = torchac_utils.estimate_bitrate_from_pmf(prob, sym=sym.long())
        # Convert to a torchac-compatible CDF.
        
        # torch.cuda.synchronize()
        # t = time.time()

        output_cdf = torchac_utils.pmf_to_cdf(prob)
        # torchac expects sym as int16, see README for details.
        sym = sym.to(torch.int16)
        # torchac expects CDF and sym on CPU.
        output_cdf = output_cdf.detach().cpu()
        sym = sym.detach().cpu()
        # Get real bitrate from the byte_stream.
        byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
        real_bits = len(byte_stream) * 8
        # print('in torchac encode: ', time.time() - t)
        
        # if _WRITE_BITS:
        #     # Write to a file.
        #     with open('outfile.b', 'wb') as fout:
        #         fout.write(byte_stream)
        #     # Read from a file.
        #     with open('outfile.b', 'rb') as fin:
        #         byte_stream = fin.read()
        # assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
        
        # t = time.time()
        # assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
        # print('torchac decode: ', time.time() - t)
        return estimated_bits, real_bits

    def bitrate_loss_all(self, prob, sym):
        prob = prob.unsqueeze(1)  # B x 1 x H x W x Q_len  # 1 is channel_num
        sym = sym.permute(0, 3, 1, 2)  # B x 1 x H x W  # range: [0, Q_len - 1]
        estimated_bits = torchac_utils.estimate_bitrate_from_pmf(prob, sym=sym.long())
        # Convert to a torchac-compatible CDF.

        output_cdf = torchac_utils.pmf_to_cdf(prob)
        # torchac expects sym as int16, see README for details.
        sym = sym.to(torch.int16)
        # torchac expects CDF and sym on CPU.
        output_cdf = output_cdf.detach().cpu()
        sym = sym.detach().cpu()
        # Get real bitrate from the byte_stream.
        byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
        real_bits = len(byte_stream) * 8
        # if _WRITE_BITS:
        #     # Write to a file.
        #     with open('outfile.b', 'wb') as fout:
        #         fout.write(byte_stream)
        #     # Read from a file.
        #     with open('outfile.b', 'rb') as fin:
        #         byte_stream = fin.read()
        # assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
        return estimated_bits, real_bits
    
    def get_triplet_loss(self, point_feats, point_class):
        bs = point_feats.shape[0]

        triplet_loss = 0
        for b in range(bs):
            IPython.embed()
            triplet_loss += self.triplet_loss(point_feats[b].view((-1, point_feats.shape[-1]))[:10], point_class[b].view((-1, 1))[:10])

        triplet_loss = triplet_loss / bs
        return triplet_loss
    
    

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (N, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (N, num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #
        # IPython.embed()
        dist = torch.norm(inputs.unsqueeze(1) - inputs.unsqueeze(0), 2, dim=-1)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        # IPython.embed()

        return self.ranking_loss(dist_an, dist_ap, y)
