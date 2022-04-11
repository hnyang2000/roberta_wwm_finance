# coding=utf-8
from torch.autograd import Variable
import torch.nn as nn
import torch


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            target_size: int, target size
            use_cuda: bool, 是否使用gpu, default is True
            average_batch: bool, loss是否作平均, default is True
        """
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        init_transitions = torch.zeros(self.target_size + 4, self.target_size + 4)
        init_transitions[:, self.START_TAG_IDX] = -1000.
        init_transitions[self.END_TAG_IDX, :] = -1000.
        # if self.use_cuda:
        #     init_transitions = init_transitions.to(DEVICE)
        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask=None):
        """
        Do the forward algorithm to compute the partition function (batched)，采用逐层log sum exp

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # 不同的位置（368个）不同的转移分数和发射分数的和的矩阵（368x11x11），例如在cls位置，cls为标记O(O,B-PER..)且从O转到O的分数
        # score矩阵是一句话（从cls到sep）的任何位置前一个标记到当前标记的发射和转移分数之和矩阵
        # 转移矩阵是前一个标记到当前标记的转移分数矩阵
        # values矩阵是一句话的任何位置从初始start标记到当前标记节点路径分数之和的矩阵
        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        # 序列的每个位置
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        # cls位置为各种标记(O,B-PER..)且从start标记转到该标记的分数
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        for idx, cur_values in seq_iter:
            # cls位置从start到各标记(例如O)的分数加上当前位置从前个标记（例如O）转到当前不同标记的分数
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            # 将当前分数转化为当前配分值
            cur_partition = log_sum_exp(cur_values, tag_size)
            # 掩掉padding
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx.byte() == 1)

            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx.byte() == 1, masked_cur_partition)
        # 序列的最后一个位置从前个标记转到当前各个标记的分数
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        # 序列的最后一个位置从之前所有标记转到当前各个标记的分数
        cur_partition = log_sum_exp(cur_values, tag_size)
        # 序列的最后一个位置从之前所有标记转到结束标记的分数
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask=None):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            # 当前位置i各个标记j所对应的分数partition和前个位置l
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size) == 1, 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
                      self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = Variable(torch.zeros(batch_size, tag_size)).long().to(feats.device)
        # if self.use_cuda:
        #     pad_zero = pad_zero.to(DEVICE)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = Variable(torch.LongTensor(seq_len, batch_size)).to(feats.device)
        # if self.use_cuda:
        #     decode_idx = decode_idx.to(DEVICE)
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask=None):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        new_tags = Variable(torch.LongTensor(batch_size, seq_len)).to(tags.device)
        # if self.use_cuda:
        #     new_tags = new_tags.to(DEVICE)
        for idx in range(seq_len):
            if idx == 0:# 第一个位置从start到cls标记的分数在score的位置的索引
                index = (tag_size - 2) * tag_size + tags[:, 0]
                new_tags[:, 0] = torch.where((index >= 0) & (index < tag_size**2), index, 0)
            else:# 任意位置从前个标记到tag里该位置的标记的分数在score里位置的索引
                index = tags[:, idx - 1] * tag_size + tags[:, idx]
                new_tags[:, idx] = torch.where((index >= 0) & (index < tag_size**2), index, 0)
        # 结束之后，再加个最后位置sep到end tag的分数
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        # 找到mask最后一个1位（即sep）所对应的标记
        end_ids = torch.gather(tags, 1, length_mask - 1)

        end_energy = torch.gather(end_transition, 1, end_ids)

        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0) == 1)

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        mask = mask.byte()
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score


    def _score_sentence2(self, feats, tags):
        pass
        # Gives the score of a provided tag sequence
        # score = torch.zeros(1)
        # tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # for i, feat in enumerate(feats):
        #     score = score + \
        #         self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        # return score
