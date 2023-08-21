import torch

def GAug(block , delta_G_e):
    # delta_G_e_aug = our_truncnorm(0, 1, delta_G_e, 0.03, mode='rvs')
    cos_dis = block.edata['weight'][('recipe', 'r-r', 'recipe')]
    m =  block.all_edges(etype='r-r')[0].shape[0]
    num_edge_drop = int(m * delta_G_e)
    aug_edge_num = m - num_edge_drop
    idx = torch.multinomial(cos_dis, aug_edge_num, replacement=False)
    block.remove_edges(idx.to('cpu'), etype='r-r')

    return block