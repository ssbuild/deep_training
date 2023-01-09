# -*- coding: utf-8 -*-
# @Time    : 2023/1/9 10:54
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch, math
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        # gold_rel = torch.cat([v["relation"] for v in targets])
        gold_rel = torch.cat([v["re"] for v in targets])
        # after masking the pad token
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        # gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        # gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        # gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        # gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])


        gold_head_start = torch.cat([v["sh"] for v in targets])
        gold_head_end = torch.cat([v["st"] for v in targets])
        gold_tail_start = torch.cat([v["oh"] for v in targets])
        gold_tail_end = torch.cat([v["ot"] for v in targets])

        if self.matcher == "avg":
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1/2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")

        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        #b,n
        num_gold_triples = [len(v["re"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(torch.split(cost,num_gold_triples, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.matcher = HungarianMatcher(loss_weight, matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes + 1)
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # if loss == "entity" and self.empty_targets(targets):
            #     pass
            # else:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
        return losses

    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["re"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        selected_pred_head_start = outputs["head_start_logits"][idx]
        selected_pred_head_end = outputs["head_end_logits"][idx]
        selected_pred_tail_start = outputs["tail_start_logits"][idx]
        selected_pred_tail_end = outputs["tail_end_logits"][idx]

        target_head_start = torch.cat([t["sh"][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat([t["st"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t["oh"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t["ot"][i] for t, (_, i) in zip(targets, indices)])



        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)
        losses = {'head_entity': 1/2*(head_start_loss + head_end_loss), "tail_entity": 1/2*(tail_start_loss + tail_end_loss)}
        # print(losses)
        return losses

    @staticmethod
    def empty_targets(targets):
        #没有关系
        flag = True
        for target in targets:
            #有关系
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag
