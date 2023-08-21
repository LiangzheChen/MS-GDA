import dgl
import dgl.nn as dglnn
from dgl.nn.pytorch import GATv2Conv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Attention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature):
        super().__init__()

        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, queries, keys, values):
        """
        It is equivariant to permutations
        of the batch dimension (`b`).

        It is equivariant to permutations of the
        second dimension of the queries (`n`).

        It is invariant to permutations of the
        second dimension of keys and values (`m`).

        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d'].
        Returns:
            a float tensor with shape [b, n, d'].
        """

        attention = torch.bmm(queries, keys.transpose(1, 2))
        attention = self.softmax(attention / self.temperature)
        # it has shape [b, n, m]

        return torch.bmm(attention, values)

class MultiheadAttention(nn.Module):

    def __init__(self, d, h):
        """
        Arguments:
            d: an integer, dimension of queries and values.
                It is assumed that input and
                output dimensions are the same.
            h: an integer, number of heads.
        """
        super().__init__()

        assert d % h == 0
        self.h = h

        # everything is projected to this dimension
        p = d // h

        self.project_queries = nn.Linear(d, d)
        self.project_keys = nn.Linear(d, d)
        self.project_values = nn.Linear(d, d)
        self.concatenation = nn.Linear(d, d)
        self.attention = Attention(temperature=p**0.5)

    def forward(self, queries, keys, values):
        """
        Arguments:
            queries: a float tensor with shape [b, n, d].
            keys: a float tensor with shape [b, m, d].
            values: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """

        h = self.h
        b, n, d = queries.size()
        _, m, _ = keys.size()
        p = d // h

        queries = self.project_queries(queries)  # shape [b, n, d]
        keys = self.project_keys(keys)  # shape [b, m, d]
        values = self.project_values(values)  # shape [b, m, d]

        queries = queries.view(b, n, h, p)
        keys = keys.view(b, m, h, p)
        values = values.view(b, m, h, p)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(h * b, n, p)
        keys = keys.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)
        values = values.permute(2, 0, 1, 3).contiguous().view(h * b, m, p)

        output = self.attention(queries, keys, values)  # shape [h * b, n, p]
        output = output.view(h, b, n, p)
        output = output.permute(1, 2, 0, 3).contiguous().view(b, n, d)
        output = self.concatenation(output)  # shape [b, n, d]

        return output

class MultiheadAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()

        self.multihead = MultiheadAttention(d, h)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.rff = rff

    def forward(self, x, y):
        """
        It is equivariant to permutations of the
        second dimension of tensor x (`n`).

        It is invariant to permutations of the
        second dimension of tensor y (`m`).

        Arguments:
            x: a float tensor with shape [b, n, d].
            y: a float tensor with shape [b, m, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        h = self.layer_norm1(x + self.multihead(x, y, y))
        return self.layer_norm2(h + self.rff(h))

class InducedSetAttentionBlock(nn.Module):

    def __init__(self, d, m, h, rff1, rff2):
        """
        Arguments:
            d: an integer, input dimension.
            m: an integer, number of inducing points.
            h: an integer, number of heads.
            rff1, rff2: modules, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab1 = MultiheadAttentionBlock(d, h, rff1)
        self.mab2 = MultiheadAttentionBlock(d, h, rff2)
        self.inducing_points = nn.Parameter(torch.randn(1, m, d))

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        b = x.size(0)
        p = self.inducing_points
        p = p.repeat([b, 1, 1])  # shape [b, m, d]
        h = self.mab1(p, x)  # shape [b, m, d]
        return self.mab2(x, h)

class RFF(nn.Module):
    """
    Row-wise FeedForward layers.
    """
    def __init__(self, d):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True),
            nn.Linear(d, d), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.layers(x)

class PoolingMultiheadAttention(nn.Module):

    def __init__(self, d, k, h, rff):
        """
        Arguments:
            d: an integer, input dimension.
            k: an integer, number of seed vectors.
            h: an integer, number of heads.
            rff: a module, row-wise feedforward layers.
                It takes a float tensor with shape [b, n, d] and
                returns a float tensor with the same shape.
        """
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)
        self.seed_vectors = nn.Parameter(torch.randn(1, k, d))

    def forward(self, z):
        """
        Arguments:
            z: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, k, d].
        """
        b = z.size(0)
        s = self.seed_vectors
        s = s.repeat([b, 1, 1])  # random seed vector: shape [b, k, d]

        output = self.mab(s, z)
        # print('PoolingMultiheadAttention', output.shape)

        return output

class SetAttentionBlock(nn.Module):

    def __init__(self, d, h, rff):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d, h, rff)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, n, d].
        Returns:
            a float tensor with shape [b, n, d].
        """
        return self.mab(x, x)

class SetTransformer(nn.Module):
    def __init__(self):
        super(SetTransformer, self).__init__()
        in_dimension = 46  # 300
        out_dimension = 128  # 600

        d = in_dimension
        m = 16  # number of inducing points
        h = 2  # number of heads
        k = 2  # number of seed vectors

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )
        self.decoder_2 = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d))
        )
        self.decoder_3 = nn.Sequential(
            SetAttentionBlock(d, h, RFF(d))
        )

        self.predictor = nn.Linear(k * d, out_dimension)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """

        x = self.encoder(x)  # shape [batch, batch_max_len, d]
        x = self.decoder(x)  # shape [batch, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)
        y = self.predictor(x)

        return y

class RelationAttention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(RelationAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out = (beta * z).sum(1)  # (N, D * K)

        return out

class GNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, n_class):
        super().__init__()

        self.num_heads = 8
        self.hid_feats = int(hid_feats / self.num_heads)
        self.out_feats = int(out_feats / self.num_heads)

        self.relation_attention = RelationAttention(in_feats)  # in_feats*self.num_heads

        self.gatconv1 = dglnn.HeteroGraphConv({
            'c-i': GATv2Conv(in_feats + n_class, self.hid_feats, num_heads=self.num_heads),
            'i-c': GATv2Conv(in_feats + n_class, self.hid_feats, num_heads=self.num_heads),
            'i-r': GATv2Conv(in_feats + n_class, self.hid_feats, num_heads=self.num_heads),
            'r-i': GATv2Conv(in_feats + n_class, self.hid_feats, num_heads=self.num_heads),
            'r-r': GATv2Conv(in_feats + n_class, self.hid_feats, num_heads=self.num_heads),
            'i-i': GATv2Conv(in_feats + n_class, self.hid_feats, num_heads=self.num_heads),
        }, aggregate='stack')  # sum

        self.gatconv2 = dglnn.HeteroGraphConv({
            'c-i': GATv2Conv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'i-c': GATv2Conv(in_feats, self.hid_feats, num_heads=self.num_heads),
            'i-r': GATv2Conv(self.hid_feats * self.num_heads, self.out_feats, num_heads=self.num_heads),
            'r-i': GATv2Conv(self.hid_feats * self.num_heads, self.out_feats, num_heads=self.num_heads),
            'r-r': GATv2Conv(self.hid_feats * self.num_heads, self.out_feats, num_heads=self.num_heads),
            'i-i': GATv2Conv(self.hid_feats * self.num_heads, self.out_feats, num_heads=self.num_heads),
        }, aggregate='stack')  # sum

        self.embedding = nn.Sequential(
            nn.Linear(self.out_feats * self.num_heads, out_feats)
        )

        self.combineSetTransformerLinear = nn.Sequential(
            nn.Linear(256, 128)
        )

    def forward(self, blocks, inputs, total_ingre_emb,PIC):
        edge_weight_0 = blocks[0].edata['weight']
        edge_weight_1 = blocks[1].edata['weight']

        h = self.gatconv1(blocks[0], inputs, edge_weight_0)
        h = {k: F.relu(v).flatten(2) for k, v in h.items()}
        h = {k: self.relation_attention(v) for k, v in h.items()}

        secondToLast_ingre = h['ingredient']
        h = self.gatconv2(blocks[-1], h, edge_weight_1)  # (h, h)
        last_ingre_and_instr = h['recipe'].flatten(2)  # [64, 2, 128]

        temp = last_ingre_and_instr[:, 1, :]
        total_ingre_emb = total_ingre_emb
        temp = torch.cat([temp, total_ingre_emb], 1)
        temp = self.combineSetTransformerLinear(temp)
        combine_the_other = torch.cat([last_ingre_and_instr[:, 0, :].unsqueeze(1), temp.unsqueeze(1),PIC.unsqueeze(1)], 1)

        h = {'recipe': self.relation_attention(combine_the_other)}
        # attention-head Weight matrix
        h = {k: self.embedding(v) for k, v in h.items()}

        return torch.squeeze(h['recipe']), secondToLast_ingre, last_ingre_and_instr

class textCNN(nn.Module):
    def __init__(self, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()

        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, 1024)) for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)

    def forward(self, x):
        con_x = [conv(x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)

        return logit

def find(tensor, values):
    return torch.nonzero(tensor[..., None] == values)

def norm(input, p=1, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

class MS_GDA(nn.Module):
    def __init__(self, path, graph, device):
        super().__init__()
        self.device = device
        self.recipe_nodes_instruction_PIC = torch.load(path + 'recipe_images.pt').to(device)

        # transform input embeddings
        self.instr_embedding = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.ingredient_embedding = nn.Sequential(
            nn.Linear(46, 128),
            nn.Tanh()
        )

        self.compound_embedding = nn.Sequential(
            nn.Linear(881, 128),
            nn.Tanh()
        )

        self.setTransformer_ = SetTransformer()
        self.gnn = GNN(128, 128, 128, graph.etypes, 9)  # 128
        self.cnn = textCNN(128, kernel_wins=[3, 4, 5], dropout_rate=0.5, num_class=128)
        self.piclin = nn.Sequential(
            nn.Linear(1024, 128)
        )

        # output transformation
        self.out = nn.Sequential(
            nn.Linear(128, 9)
        )

        self.dropout = nn.Dropout(0.75)

        self.ingre_neighbor_tensor, self.ingre_length_tensor, self.total_length_index_list, total_ingre_neighbor_tensor = self.get_recipe2ingreNeighbor_dict(graph)


    def get_ingredient_neighbors_link_scores(self, blocks, output_nodes, secondToLast_ingre, recipe):
        ingreNodeIDs = blocks[1].srcdata['_ID']['ingredient']
        recipeNodeIDs = output_nodes['recipe']
        batch_ingre_neighbors = self.ingre_neighbor_tensor[recipeNodeIDs].to(self.device)
        batch_ingre_length = self.ingre_length_tensor[recipeNodeIDs]
        valid_batch_ingre_neighbors = find(batch_ingre_neighbors, ingreNodeIDs)[:, 2]

        # based on valid_batch_ingre_neighbors each row index
        _, valid_batch_ingre_length = torch.unique(find(batch_ingre_neighbors, ingreNodeIDs)[:, 0], return_counts=True)
        batch_sum_ingre_length = np.cumsum(valid_batch_ingre_length.cpu())

        total_ingre_emb = None
        total_pos_score = None
        total_neg_score = None

        for i in range(len(recipeNodeIDs)):
            if i == 0:
                recipeNode_ingres = valid_batch_ingre_neighbors[0:batch_sum_ingre_length[i]]
                potential_neg_ingres = valid_batch_ingre_neighbors[batch_sum_ingre_length[i]:]
                neg_ingres = potential_neg_ingres[torch.randint(len(potential_neg_ingres), (len(recipeNode_ingres),))]
                a = secondToLast_ingre[recipeNode_ingres]
                b = secondToLast_ingre[neg_ingres]
            else:
                recipeNode_ingres = valid_batch_ingre_neighbors[batch_sum_ingre_length[i - 1]:batch_sum_ingre_length[i]]
                potential_neg_ingres = torch.cat([valid_batch_ingre_neighbors[:batch_sum_ingre_length[i - 1]],
                                                  valid_batch_ingre_neighbors[batch_sum_ingre_length[i]:]])
                neg_ingres = potential_neg_ingres[torch.randint(len(potential_neg_ingres), (len(recipeNode_ingres),))]
                a = secondToLast_ingre[recipeNode_ingres]
                b = secondToLast_ingre[neg_ingres]

            cur_recipe = recipe[i, :]
            pos_score = torch.mm(a, cur_recipe.unsqueeze(1))
            neg_score = torch.mm(b, cur_recipe.unsqueeze(1))

            if total_pos_score == None:
                total_pos_score = pos_score
                total_neg_score = neg_score
            else:
                total_pos_score = torch.cat([total_pos_score, pos_score], dim=0)
                total_neg_score = torch.cat([total_neg_score, neg_score], dim=0)

        total_pos_score = total_pos_score.squeeze()
        total_neg_score = total_neg_score.squeeze()

        return total_pos_score, total_neg_score

    def get_recipe2ingreNeighbor_dict(self, graph):
        max_length = 33
        # print(max(len(x) for x in neighbor_list))
        out = {}
        neighbor_list = []
        ingre_length_list = []
        total_length_index_list = []
        total_ingre_neighbor_list = []
        total_length_index = 0
        total_length_index_list.append(total_length_index)
        for recipeNodeID in tqdm(range(graph.number_of_nodes('recipe'))):
            _, succs = graph.out_edges(recipeNodeID, etype='r-i')
            succs_list = list(set(succs.tolist()))
            total_ingre_neighbor_list.extend(succs_list)
            cur_length = len(succs_list)
            ingre_length_list.append(cur_length)

            total_length_index += cur_length
            total_length_index_list.append(total_length_index)
            while len(succs_list) < max_length:
                succs_list.append(77733)
            neighbor_list.append(succs_list)

        ingre_neighbor_tensor = torch.tensor(neighbor_list)
        ingre_length_tensor = torch.tensor(ingre_length_list)
        total_ingre_neighbor_tensor = torch.tensor(total_ingre_neighbor_list)
        return ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, total_ingre_neighbor_tensor

    def get_ingredient_neighbors_all_embeddings(self, blocks, output_nodes, secondToLast_ingre):
        ingreNodeIDs = blocks[1].srcdata['_ID']['ingredient']
        recipeNodeIDs = output_nodes['recipe']
        batch_ingre_neighbors = self.ingre_neighbor_tensor[recipeNodeIDs].to(self.device)
        batch_ingre_length = self.ingre_length_tensor[recipeNodeIDs]
        valid_batch_ingre_neighbors = find(batch_ingre_neighbors, ingreNodeIDs)[:, 2]

        # based on valid_batch_ingre_neighbors each row index
        _, valid_batch_ingre_length = torch.unique(find(batch_ingre_neighbors, ingreNodeIDs)[:, 0], return_counts=True)
        batch_sum_ingre_length = np.cumsum(valid_batch_ingre_length.cpu())

        total_ingre_emb = None
        for i in range(len(recipeNodeIDs)):
            if i == 0:
                recipeNode_ingres = valid_batch_ingre_neighbors[0:batch_sum_ingre_length[i]]
                a = secondToLast_ingre[recipeNode_ingres]
            else:
                recipeNode_ingres = valid_batch_ingre_neighbors[batch_sum_ingre_length[i - 1]:batch_sum_ingre_length[i]]
                a = secondToLast_ingre[recipeNode_ingres]

            # all ingre instead of average
            a_rows = a.shape[0]
            a_columns = a.shape[1]
            max_rows = 5
            if a_rows < max_rows:
                a = torch.cat([a, torch.zeros(max_rows - a_rows, a_columns).cuda()])
            else:
                a = a[:max_rows, :]

            if total_ingre_emb == None:
                total_ingre_emb = a.unsqueeze(0)
            else:
                total_ingre_emb = torch.cat([total_ingre_emb, a.unsqueeze(0)], dim=0)
                if torch.isnan(total_ingre_emb).any():
                    print('Error!')

        return total_ingre_emb

    def forward(self, graph, inputs, output_nodes, la_emb):
        instr, ingredient, compound, ingredient_of_dst_recipe = inputs

        # instruction - textcnn
        # PIC = instr[:, 20, :]
        # instr = instr[:,0:20,:]

        instr = self.cnn(instr.unsqueeze(1))
        instr = norm(instr)

        PICIDS = output_nodes['recipe']
        # PICIDS =  blocks[1].dstdata['_ID']['recipe']
        PIC = self.recipe_nodes_instruction_PIC[PICIDS]
        PIC = self.piclin(PIC)
        PIC = norm(PIC)


        # ingredient
        ingredient = self.ingredient_embedding(ingredient)
        ingredient = norm(ingredient)
        n_classes = 9
        tampfeas = torch.zeros([ingredient.shape[0], int(n_classes)],dtype=torch.float32).to(self.device)
        ingredient = torch.cat([ingredient, tampfeas], dim=1)


        la_emb[0:output_nodes['recipe'].shape[0]] = torch.zeros([output_nodes['recipe'].shape[0],la_emb.shape[1]],dtype=torch.float32).to(self.device)
        instr = torch.cat([instr,la_emb],dim=1)
        instr = self.dropout(instr)
        #compound
        compound = self.compound_embedding(compound)
        compound = norm(compound)

        tampfeas = torch.zeros([compound.shape[0], int(n_classes)], dtype=torch.float32).to(self.device)
        compound = torch.cat([compound, tampfeas], dim=1)
        # for setTransformer
        all_ingre_emb_for_each_recipe = self.get_ingredient_neighbors_all_embeddings(graph, output_nodes,
                                                                                ingredient_of_dst_recipe)
        all_ingre_emb_for_each_recipe = norm(all_ingre_emb_for_each_recipe)

        total_ingre_emb = self.setTransformer_(all_ingre_emb_for_each_recipe)
        # if total_ingre_emb.shape[0]!=64:
        #     print(1)
        # GNN
        output, secondToLast_ingre, last_ingre_and_instr = self.gnn(graph, {'recipe': instr, 'ingredient': ingredient,'compound': compound },
                                                                    total_ingre_emb,PIC)
        total_pos_score, total_neg_score = self.get_ingredient_neighbors_link_scores(graph, output_nodes, secondToLast_ingre,
                                                                                output)

        return self.out(output), secondToLast_ingre, output, total_pos_score, total_neg_score