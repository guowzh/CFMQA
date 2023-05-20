from numpy.core.numeric import full
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.autograd import Variable

from transformers.modeling_bert import BertLayer, gelu, BertConfig
from csr_mhqa.utils import get_weights, get_act


def tok_to_ent(tok2ent):
    if tok2ent == 'mean':
        return MeanPooling
    elif tok2ent == 'mean_max':
        return MeanMaxPooling
    else:
        raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x


def mean_pooling(input, mask):
    mean_pooled = input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    return mean_pooled


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens):
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        return mean_pooled

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, doc_state, entity_mapping, entity_lens=None):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
        if entity_lens is None:
            entity_lens = entity_mapping.sum(-1)
            entity_lens = entity_lens + (entity_lens == 0)
        mean_pooled = torch.sum(entity_states, dim=2) / entity_lens.unsqueeze(2)
        output = torch.cat([max_pooled, mean_pooled], dim=2)  # N x E x 2d
        return output

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class GATSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, config, q_attn=False, head_id=0):
        """ One head GAT """
        super(GATSelfAttention, self).__init__()
        self.config = config
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = self.config.gnn_drop
        self.q_attn = q_attn
        self.query_dim = in_dim
        self.n_type = self.config.num_edge_type

        self.head_id = head_id
        self.step = 0

        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.qattn_W1 = nn.ParameterList()
        self.qattn_W2 = nn.ParameterList()
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_dim, out_dim)))
            self.a_type.append(get_weights((out_dim * 2, 1)))
            
            if self.q_attn:
                q_dim = self.config.hidden_dim if self.config.q_update else self.config.input_dim
                self.qattn_W1.append(get_weights((q_dim, out_dim * 2)))
                self.qattn_W2.append(get_weights((out_dim * 2, out_dim * 2)))

        self.act = get_act('lrelu:0.2')

    def forward(self, input_state, adj, node_mask=None, query_vec=None):
        zero_vec = torch.zeros_like(adj)
        scores = torch.zeros_like(adj)

        for i in range(self.n_type):
            h = torch.matmul(input_state, self.W_type[i])
            h = F.dropout(h, self.dropout, self.training)
            N, E, d = h.shape

            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)  #repeat n_nodes embedding
            a_input = a_input.view(-1, E, E, 2*d)

            if self.q_attn:
                q_gate = F.relu(torch.matmul(query_vec, self.qattn_W1[i]))
                q_gate = torch.sigmoid(torch.matmul(q_gate, self.qattn_W2[i]))
                a_input = a_input * q_gate[:, None, None, :]
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))
            else:
                score = self.act(torch.matmul(a_input, self.a_type[i]).squeeze(3))

            scores += torch.where(adj == i+1, score, zero_vec.to(score.dtype))

        zero_vec = -1e30 * torch.ones_like(scores)
        scores = torch.where(adj > 0, scores, zero_vec.to(scores.dtype))

        # Ahead Alloc
        if node_mask is not None:
            h = h * node_mask

        coefs = F.softmax(scores, dim=2)  # N * E * E
        # h = coefs.unsqueeze(3) * h.unsqueeze(2)  # N * E * E * d    #this equals h = torch.matmul(scores.permute(0,2,1),h)
        # h = torch.sum(h, dim=1)                                    #???it should be h = torch.matmul(coefs,h)
        h = torch.matmul(coefs,h)
        return h

class FullyConnectedGATSelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim, config, q_attn=False, head_id=0):
        super(FullyConnectedGATSelfAttention,self).__init__()
        self.config = config
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = self.config.gnn_drop
        self.q_attn = q_attn
        self.query_dim = in_dim
        self.n_type = self.config.num_edge_type

        self.head_id = head_id
        self.step = 0

        self.W = nn.Linear(in_dim,out_dim)
        self.a = nn.Linear(out_dim*2,1)
        if self.q_attn:
            q_dim = self.config.hidden_dim if self.config.q_update else self.config.input_dim
            self.qattn_W1 = nn.Linear(q_dim,out_dim*2)
            self.qattn_W2 = nn.Linear(out_dim*2,out_dim*2)
        self.act = get_act("lrelu:0.2")

    def forward(self, input_state, adj, node_mask=None, query_vec=None):
        """
        input: b x 105 x d
        adj: b x 105 x 105
        node_mask: b x 105 x 1
        query_vec: b x encoder_dim
        """
        h = self.W(input_state)
        h = F.dropout(h,self.dropout,self.training)
        N,E,d = h.shape

        a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim=-1)
        a_input = a_input.view(-1, E, E, 2*d)
        if self.q_attn:
            q_gate = F.relu(self.qattn_W1(query_vec))
            q_gate = torch.sigmoid(self.qattn_W2(q_gate))
            a_input = a_input * q_gate[:,None,None,:]
            scores = self.act(self.a(a_input).squeeze(3))
        else:
            scores = self.act(self.a(a_input).squeeze(3))
        
        fullyconnected_adj = torch.ones_like(adj).to(scores.dtype)
        fullyconnected_adj[node_mask.squeeze(-1)==0] = 0
        fullyconnected_adj = fullyconnected_adj * fullyconnected_adj.transpose(2,1)
        zero_vec = -1e30 * torch.ones_like(scores).to(scores.dtype)
        scores = torch.where(fullyconnected_adj > 0, scores,zero_vec)
        if node_mask is not None:
            h = h * node_mask
        coefs = F.softmax(scores,dim=-1)
        h = torch.matmul(coefs,h)
        return h

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_head, q_attn, config):
        super(AttentionLayer, self).__init__()
        assert hid_dim % n_head == 0
        self.dropout = config.gnn_drop

        self.attn_funcs = nn.ModuleList()
        for i in range(n_head):
            self.attn_funcs.append(
                    GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, q_attn=q_attn, head_id=i))

        if in_dim != hid_dim:
            self.align_dim = nn.Linear(in_dim, hid_dim)
            nn.init.xavier_uniform_(self.align_dim.weight, gain=1.414)
        else:
            self.align_dim = lambda x: x

    def forward(self, input, adj, node_mask=None, query_vec=None):
        hidden_list = []
        for attn in self.attn_funcs:
            h = attn(input, adj, node_mask=node_mask, query_vec=query_vec)
            hidden_list.append(h)

        h = torch.cat(hidden_list, dim=-1)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        return h


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, config, num_answer=1):
        super(OutputLayer, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            BertLayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(config.trans_drop),
            nn.Linear(hidden_dim*2, num_answer),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)


class GraphBlock(nn.Module):
    def __init__(self, q_attn, config):
        super(GraphBlock, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        if self.config.q_update:
            self.gat_linear = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        else:
            self.gat_linear = nn.Linear(self.config.input_dim, self.hidden_dim*2)

        if self.config.q_update:
            self.gat = AttentionLayer(self.hidden_dim, self.hidden_dim, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        else:
            self.gat = AttentionLayer(self.hidden_dim*2, self.hidden_dim*2, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            self.sent_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)

    def forward(self, batch, input_state, query_vec):
        context_lens = batch['context_lens']
        context_mask = batch['context_mask']
        sent_mapping = batch['sent_mapping']
        sent_start_mapping = batch['sent_start_mapping']
        sent_end_mapping = batch['sent_end_mapping']
        para_mapping = batch['para_mapping']
        para_start_mapping = batch['para_start_mapping']
        para_end_mapping = batch['para_end_mapping']
        ent_mapping = batch['ent_mapping']
        ent_start_mapping = batch['ent_start_mapping']
        ent_end_mapping = batch['ent_end_mapping']

        def get_span_pooled_vec(state_input, mapping):
            mapping_state = state_input.unsqueeze(2) * mapping.unsqueeze(3)
            mapping_sum = mapping.sum(dim=1)
            mapping_sum = torch.where(mapping_sum == 0, torch.ones_like(mapping_sum), mapping_sum)
            mean_pooled = mapping_state.sum(dim=1) / mapping_sum.unsqueeze(-1)

            return mean_pooled

        para_start_output = torch.bmm(para_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_para x d
        para_end_output = torch.bmm(para_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_para x d
        para_state = torch.cat([para_start_output, para_end_output], dim=-1)  # N x max_para x 2d

        sent_start_output = torch.bmm(sent_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_sent x d
        sent_end_output = torch.bmm(sent_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_sent x d
        sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d

        ent_start_output = torch.bmm(ent_start_mapping, input_state[:, :, self.hidden_dim:])   # N x max_ent x d
        ent_end_output = torch.bmm(ent_end_mapping, input_state[:, :, :self.hidden_dim])       # N x max_ent x d
        ent_state = torch.cat([ent_start_output, ent_end_output], dim=-1)  # N x max_ent x 2d

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        if self.config.q_update:
            graph_state = self.gat_linear(torch.cat([para_state, sent_state, ent_state], dim=1)) # N * (max_para + max_sent + max_ent) * d
            graph_state = torch.cat([query_vec.unsqueeze(1), graph_state], dim=1)
        else:
            graph_state = self.gat_linear(query_vec)
            graph_state = torch.cat([graph_state.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
        node_mask = torch.cat([torch.ones(N, 1).to(self.config.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)

        graph_adj = batch['graphs']
        assert graph_adj.size(1) == node_mask.size(1)

        graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x d
        ent_state = graph_state[:, 1+max_para_num+max_sent_num:, :]

        gat_logit = self.sent_mlp(graph_state[:, :1+max_para_num+max_sent_num, :]) # N x max_sent x 1
        para_logit = gat_logit[:, 1:1+max_para_num, :].contiguous()
        sent_logit = gat_logit[:, 1+max_para_num:, :].contiguous()

        query_vec = graph_state[:, 0, :].squeeze(1)

        ent_logit = self.entity_mlp(ent_state).view(N, -1)
        ent_logit = ent_logit - 1e30 * (1 - batch['ans_cand_mask'])

        para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
        para_prediction = torch.cat([para_logits_aux, para_logit], dim=-1).contiguous()

        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()

        return input_state, graph_state, node_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit

class GatedAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout, gate_method='gate_att_up',graph_scale=False):
        super(GatedAttention, self).__init__()
        self.gate_method = gate_method
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_1 = nn.Linear(memory_dim, hid_dim, bias=True)

        out_dim = hid_dim if not graph_scale else hid_dim*2
        self.input_linear_2 = nn.Linear(input_dim + memory_dim, out_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask,ans_predictions=None):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input_dot = F.relu(self.input_linear_1(input))  # N x Ld x d
        memory_dot = F.relu(self.memory_linear_1(memory))  # N x Lm x d

        # N * Ld * Lm
        att = torch.bmm(input_dot, memory_dot.permute(0, 2, 1).contiguous()) / self.dot_scale
        if ans_predictions is not None:
            att = att * ans_predictions.unsqueeze(1)

        att = att - 1e30 * (1 - mask[:, None])
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)

        if self.gate_method == 'no_gate':
            output = torch.cat( [input, output_one], dim=-1 )
            output = F.relu(self.input_linear_2(output))
        elif self.gate_method == 'gate_att_or':
            output = torch.cat( [input, input - output_one], dim=-1) 
            output = F.relu(self.input_linear_2(output))
        elif self.gate_method == 'gate_att_up':
            output = torch.cat([input, output_one], dim=-1 )
            gate_sg = torch.sigmoid(self.input_linear_2(output))
            gate_th = torch.tanh(self.input_linear_2(output))
            output = gate_sg * gate_th
        else:
            raise ValueError("Not support gate method: {}".format(self.gate_method))


        return output, memory


class BiAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask,only_c2q=False):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = F.dropout(input, self.dropout, training=self.training)  # N x Ld x d
        memory = F.dropout(memory, self.dropout, training=self.training)  # N x Lm x d

        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [f1, f2]^T [w1, w2] + <f1 * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm
        att = att - 1e30 * (1 - mask[:, None])

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F.softmax(att, dim=-1)         # N x M similarity matrix
        output_one = torch.bmm(weight_one, memory)  #context2query   N x d
        if only_c2q:
            return output_one,memory
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)  # 1 x N
        output_two = torch.bmm(weight_two, input)   #query2context 1 x N N x d -> 1 x d

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), memory


class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config, h_dim=None):
        super(PredictionLayer, self).__init__()
        self.config = config
        input_dim = config.ctx_attn_hidden_dim if h_dim is None else h_dim

        self.start_linear = OutputLayer(input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(input_dim, config, num_answer=4)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        # context_lens = batch['context_lens']
        # sent_mapping = batch['sent_mapping']

        # sp_forward = torch.bmm(sent_mapping, sent_logits).contiguous()  # N x max_seq_len x 1

        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])

        if not return_yp:
            return start_prediction, end_prediction, type_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)    #start position < end position
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, type_prediction, yp1, yp2


class InteractionLayer(nn.Module):
    def __init__(self, input_dim, out_dim, config):
        super(InteractionLayer, self).__init__()
        self.config = config
        self.use_trans = config.basicblock_trans

        if config.basicblock_trans:
            bert_config = BertConfig(input_dim, config.trans_heads, config.trans_drop)
            self.transformer = BertLayer(bert_config)
            self.transformer_linear = nn.Linear(input_dim, out_dim)
        else:
            self.lstm = LSTMWrapper(input_dim, out_dim // 2, 1)

    def forward(self, doc_state, entity_state, doc_length, entity_mapping, entity_length, context_mask):
        """
        :param doc_state: N x L x dc
        :param entity_state: N x E x de
        :param entity_mapping: N x E x L
        :return: doc_state: N x L x out_dim, entity_state: N x L x out_dim (x2)
        """
        expand_entity_state = torch.sum(entity_state.unsqueeze(2) * entity_mapping.unsqueeze(3), dim=1)  # N x E x L x d
        input_state = torch.cat([expand_entity_state, doc_state], dim=2)

        if self.use_trans:
            extended_attention_mask = context_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            doc_state = self.transformer(input_state, extended_attention_mask)
            doc_state = self.transformer_linear(doc_state)
        else:
            doc_state = self.lstm(input_state, doc_length)

        return doc_state

class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, q_dim, layer, config):
        super(BasicBlock, self).__init__()
        self.config = config
        self.layer = layer
        self.gnn_type = config.gnn.split(':')[0]
        if config.tok2ent == 'mean_max':
            input_dim = hidden_dim * 2
        else:
            input_dim = hidden_dim
        self.tok2ent = tok_to_ent(config.tok2ent)()
        self.query_weight = get_weights((q_dim, input_dim))
        self.temp = np.sqrt(q_dim * input_dim)
        self.gat = AttentionLayer(input_dim, hidden_dim, config.num_gnn_heads, config.q_attn ,config)
        self.int_layer = InteractionLayer(hidden_dim * 2, hidden_dim, config)

    def forward(self, doc_state, query_vec, batch):
        context_mask = batch['context_mask']
        entity_mapping = batch['ent_mapping'].permute(0,2,1)
        entity_mask = batch['ent_mask']
        doc_length = batch['context_lens']
        max_entity = entity_mask.shape[-1]
        adj = batch['graphs'][:,-max_entity:,-max_entity:]

        entity_length = entity_mapping.sum(-1)
        entity_length = entity_length + (entity_length == 0)

        entity_state = self.tok2ent(doc_state, entity_mapping, entity_length)

        query = torch.matmul(query_vec, self.query_weight)
        query_scores = torch.bmm(entity_state, query.unsqueeze(2)) / self.temp
        softmask = query_scores * entity_mask.unsqueeze(2)  # N x E x 1  BCELossWithLogits
        adj_mask = torch.sigmoid(softmask)

        entity_state = self.gat(entity_state, adj, adj_mask, query_vec=query_vec)
        doc_state = self.int_layer(doc_state, entity_state, doc_length, entity_mapping, entity_length, context_mask)
        return doc_state, entity_state, softmask

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    Sinusoid position encoding table
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class PositionalEncoder(nn.Module):
    def __init__(self, h_dim, config):
        super(PositionalEncoder, self).__init__()
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(config.max_doc_len, h_dim, padding_idx=0),
            freeze=True)
        self.device = 'cuda:{}'.format(config.model_gpu)
        self.fixed_position_vec = torch.LongTensor(list(range(config.max_doc_len))).cuda(self.device)

    def forward(self, context_mapping):
        """
        :param context_mapping: N x L
        :return: position_encoding: N x L x d
        """
        N, L = context_mapping.shape
        trunc_position_vec = self.fixed_position_vec[:L].contiguous()
        context_position = trunc_position_vec.unsqueeze(0) * context_mapping
        position_encoding = self.position_enc(context_position)
        return position_encoding

class TransformerPredictionLayer(nn.Module):
    def __init__(self, config, q_dim):
        super(TransformerPredictionLayer, self).__init__()
        self.config = config
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.position_encoder = PositionalEncoder(h_dim, config)

        # Cascade Network
        bert_config = BertConfig(config.hidden_dim, config.trans_heads, config.trans_drop)

        self.sp_transformer = BertLayer(bert_config)
        self.sp_linear = nn.Linear(h_dim * 2, 1)

        self.start_input_linear = nn.Linear(h_dim + 1, h_dim)
        self.start_transformer = BertLayer(bert_config)
        self.start_linear = nn.Linear(h_dim, 1)

        self.end_input_linear = nn.Linear(2 * h_dim + 1, h_dim)
        self.end_transformer = BertLayer(bert_config)
        self.end_linear = nn.Linear(h_dim, 1)

        self.type_input_linear = nn.Linear(2 * h_dim + 1, h_dim)
        self.type_transformer = BertLayer(bert_config)
        self.type_linear = nn.Linear(h_dim, 3)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, query_vec, entity_state, packing_mask=None, return_yp=False):
        """
        :param context_input:   [N x L x hid]
        :param query_vec:       [N x q_dim]
        :param entity_state:    [N x E x hid]
        :param entity_mask:     [N x E]
        :param context_mask:    [N x L]
        :param context_lens:    [N]
        :param start_mapping:   [N x max_sent x L]      000100000000
        :param end_mapping:     [N x max_sent x L]      000000001000
        :param all_mapping:     [N x L x max_sent]      000111111000
        :param packing_mask     [N x L]
        :param return_yp:       bool
        :return:
        """
        context_mask = batch['context_mask']
        start_mapping = batch['start_mapping']
        end_mapping = batch['end_mapping']
        all_mapping = batch['all_mapping']

        position_encoding = self.position_encoder(context_mask.long())

        extended_attention_mask = context_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        entity_prediction = None

        context_input = context_input + position_encoding
        sp_output = self.sp_transformer(context_input, extended_attention_mask)  # N x L x d
        sp_start_output = torch.bmm(start_mapping, sp_output)   # N x max_sent x d
        sp_end_output = torch.bmm(end_mapping, sp_output)       # N x max_sent x d
        sp_logits = torch.cat([sp_start_output, sp_end_output], dim=-1)  # N x max_sent x 2d

        sp_logits = self.sp_linear(sp_logits)  # N x max_sent x 1
        # sp_prediction = sp_logits.squeeze()
        sp_logits_aux = Variable(sp_logits.data.new(sp_logits.size(0), sp_logits.size(1), 1).zero_())
        sp_prediction = torch.cat([sp_logits_aux, sp_logits], dim=-1).contiguous()
        sp_forward = torch.bmm(all_mapping, sp_logits).contiguous()  # N x L x 1

        start_input = torch.cat([context_input, sp_forward], dim=-1)
        start_input = self.start_input_linear(start_input) + position_encoding
        start_output = self.start_transformer(start_input, extended_attention_mask)
        start_prediction = self.start_linear(start_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        end_input = torch.cat([context_input, start_output, sp_forward], dim=-1)
        end_input = self.end_input_linear(end_input) + position_encoding
        end_output = self.end_transformer(end_input, extended_attention_mask)
        end_prediction = self.end_linear(end_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        type_input = torch.cat([context_input, end_output, sp_forward], dim=-1)
        type_input = self.type_input_linear(type_input) + position_encoding
        type_output = torch.max(self.type_transformer(type_input, extended_attention_mask), dim=1)[0]
        type_logits = type_output
        type_prediction = self.type_linear(type_logits)

        if not return_yp:
            return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction, yp1, yp2

class DFGNPredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config, q_dim):
        super(DFGNPredictionLayer, self).__init__()
        self.config = config
        input_dim = config.hidden_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        # Cascade Network
        self.entity_linear_0 = nn.Linear(h_dim + q_dim, h_dim)
        self.entity_linear_1 = nn.Linear(h_dim, 1)

        self.sp_lstm = LSTMWrapper(input_dim=input_dim, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.sp_linear = nn.Linear(h_dim * 2, 1)

        self.start_lstm = LSTMWrapper(input_dim=input_dim + 1, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.start_linear = nn.Linear(h_dim * 2, 1)

        self.end_lstm = LSTMWrapper(input_dim=input_dim + 2*h_dim + 1, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.end_linear = nn.Linear(h_dim * 2, 1)

        self.type_lstm = LSTMWrapper(input_dim=input_dim + 2*h_dim + 1, hidden_dim=h_dim, n_layer=1, dropout=config.lstm_drop)
        self.type_linear = nn.Linear(h_dim * 2, 4)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, query_vec, entity_state, packing_mask=None, return_yp=False):
        """
        :param batch
        :param context_input:   [N x L x hid]
        :param query_vec:       [N x q_dim]
        :param entity_state:    [N x E x hid]
        :param entity_mask:     [N x E]
        :param context_mask:    [N x L]
        :param context_lens:    [N]
        :param start_mapping:   [N x max_sent x L]      000100000000
        :param end_mapping:     [N x max_sent x L]      000000001000
        :param all_mapping:     [N x L x max_sent]      000111111000
        :param packing_mask:    [N x L] or None
        :param return_yp
        :return:
        """
        context_mask = batch['context_mask']
        entity_mask = batch['ent_mask']
        context_lens = batch['context_lens']
        start_mapping = batch['sent_start_mapping']
        end_mapping = batch['sent_end_mapping']
        all_mapping = batch['sent_mapping']

        entity_prediction = None
        if entity_state is not None:
            expand_query = query_vec.unsqueeze(1).repeat((1, entity_state.shape[1], 1))
            entity_logits = self.entity_linear_0(torch.cat([entity_state, expand_query], dim=2))
            entity_logits = self.entity_linear_1(F.relu(entity_logits))
            entity_prediction = entity_logits.squeeze(2) - 1e30 * (1 - entity_mask)

        sp_output = self.sp_lstm(context_input, context_lens)  # N x L x 2d
        start_output = torch.bmm(start_mapping, sp_output[:, :, self.hidden:])   # N x max_sent x d
        end_output = torch.bmm(end_mapping, sp_output[:, :, :self.hidden])       # N x max_sent x d
        sp_logits = torch.cat([start_output, end_output], dim=-1)  # N x max_sent x 2d

        sp_logits = self.sp_linear(sp_logits)  # N x max_sent x 1
        sp_logits_aux = Variable(sp_logits.data.new(sp_logits.size(0), sp_logits.size(1), 1).zero_())
        sp_prediction = torch.cat([sp_logits_aux, sp_logits], dim=-1).contiguous()

        sp_forward = torch.bmm(all_mapping, sp_logits).contiguous()  # N x L x 1

        start_input = torch.cat([context_input, sp_forward], dim=-1)
        start_output = self.start_lstm(start_input, context_lens)
        start_prediction = self.start_linear(start_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        end_input = torch.cat([context_input, start_output, sp_forward], dim=-1)
        end_output = self.end_lstm(end_input, context_lens)
        end_prediction = self.end_linear(end_output).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        type_input = torch.cat([context_input, end_output, sp_forward], dim=-1)
        type_output = torch.max(self.type_lstm(type_input, context_lens), dim=1)[0]
        type_logits = type_output
        type_prediction = self.type_linear(type_logits)

        if not return_yp:
            return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, sp_prediction, type_prediction, entity_prediction, yp1, yp2