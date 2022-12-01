import torch.nn as nn
import torch

def generate_square_subsequent_mask(sz: int, device='cpu'):
    
    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def tgt_mask(lengths, max_len=None):

    sqr_subseq_mask = torch.tril(torch.ones(max_len, max_len) == 1)
    pad_mask = torch.arange(0, max_len).repeat(max_len, 1).lt(lengths.unsqueeze(1).unsqueeze(1))

    mask = pad_mask & sqr_subseq_mask
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf]]

    return mask

def src_mask(lengths, max_len=None):

    mask = ~torch.arange(0, max_len).repeat(max_len, 1).lt(lengths.unsqueeze(1).unsqueeze(1))
    mask = mask.float()
    mask = mask.masked_fill(mask == 1, float('-inf'))

    return mask

if __name__ == "__main__":

    attn_mask = generate_square_subsequent_mask(8)
    print('attn_mask ', attn_mask)

    lengths = torch.tensor([2, 3, 6])
    srcmask = src_mask(lengths, max_len=6)
    print('srcmask ', srcmask, srcmask.shape)

    # print('lengths ', lengths, lengths.shape)
    # pad_mask = padding_mask(lengths, 36)
    # print('pad_mask ', pad_mask)

    # inputs = torch.ones((8, 2, 6))
    # mha = torch.nn.MultiheadAttention(6, 2) # hidden_dim=6, head_num=2
    # outputs, weights = mha(inputs, inputs, inputs, attn_mask=attn_mask)
    # print('outputs ', outputs)
    # print('weights ', weights)