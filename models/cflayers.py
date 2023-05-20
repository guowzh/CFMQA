import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.autograd import Variable

from transformers.modeling_bert import BertLayer, gelu
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

    def forward(self, doc_state, entity_mapping, entity_lens):
        """
        :param doc_state:  N x L x d
        :param entity_mapping:  N x E x L
        :param entity_lens:  N x E
        :return: N x E x 2d
        """
        entity_states = entity_mapping.unsqueeze(3) * doc_state.unsqueeze(1)  # N x E x L x d
        max_pooled = torch.max(entity_states, dim=2)[0]
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
            if config.graph == "gat":
                self.attn_funcs.append(
                    GATSelfAttention(in_dim=in_dim, out_dim=hid_dim // n_head, config=config, q_attn=q_attn, head_id=i))
            elif config.graph == "fullyconnected":
                self.attn_funcs.append(
                    FullyConnectedGATSelfAttention(in_dim=in_dim,out_dim=hid_dim // n_head,config=config,q_attn=q_attn,head_id=i)
                )

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
    def __init__(self, hidden_dim, config, num_answer=1,bias=True):
        super(OutputLayer, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2,bias),
            nn.ReLU(),
            BertLayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(config.trans_drop),
            nn.Linear(hidden_dim*2, num_answer,bias),
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
            if config.graph == "gat" or config.graph == "fullyconnected":
                self.gat = AttentionLayer(self.hidden_dim, self.hidden_dim, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            elif config.graph == "without_graph":
                self.gat = lambda *args,**kwargs : args[0]
            self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        else:
            if config.graph == "gat" or config.graph == "fullyconnected":
                self.gat = AttentionLayer(self.hidden_dim*2, self.hidden_dim*2, config.num_gnn_heads, q_attn=q_attn, config=self.config)
            elif config.graph == "without_graph":
                self.gat = lambda *args,**kwargs : args[0]
            self.sent_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)
            self.entity_mlp = OutputLayer(self.hidden_dim*2, config, num_answer=1)

    def forward(self, batch, input_state, query_vec,trunc_query_state,trunc_query_mask):
        ent_start_mapping = batch['ent_start_mapping']

        def get_span_pooled_vec(state_input, mapping):
            mapping_state = state_input.unsqueeze(2) * mapping.unsqueeze(3)
            mapping_sum = mapping.sum(dim=1)
            mapping_sum = torch.where(mapping_sum == 0, torch.ones_like(mapping_sum), mapping_sum)
            mean_pooled = mapping_state.sum(dim=1) / mapping_sum.unsqueeze(-1)

            return mean_pooled
        para_state,sent_state = self.para_sent_encode(batch,input_state,trunc_query_state,trunc_query_mask)

        ent_state = torch.bmm(ent_start_mapping, input_state[:, :,:])   # N x max_ent x d

        N, max_para_num, _ = para_state.size()
        _, max_sent_num, _ = sent_state.size()
        _, max_ent_num, _ = ent_state.size()

        if self.config.q_update:
            graph_state = self.gat_linear(torch.cat([para_state, sent_state, ent_state], dim=1)) # N * (max_para + max_sent + max_ent) * d
            graph_state = torch.cat([query_vec.unsqueeze(1), graph_state], dim=1)
        else:
            q_state = self.gat_linear(query_vec)
            graph_state = torch.cat([q_state.unsqueeze(1), para_state, sent_state, ent_state], dim=1)
        node_mask = torch.cat([torch.ones(N, 1).to(self.config.device), batch['para_mask'], batch['sent_mask'], batch['ent_mask']], dim=-1).unsqueeze(-1)

        graph_adj = batch['graphs']
        assert graph_adj.size(1) == node_mask.size(1)

        graph_state = self.gat(graph_state, graph_adj, node_mask=node_mask, query_vec=query_vec) # N x (1+max_para+max_sent) x 2d
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

class AdaptedGatedAttentionWithoutqkv(nn.Module):
    def __init__(self, config, gate_method='gate_att_up'):
        super(AdaptedGatedAttentionWithoutqkv, self).__init__()
        self.gate_method = gate_method
        if gate_method not in ['gate_att_up','no_gate']:
            raise ValueError("Not support gate method: {}".format(self.gate_method))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.input_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.concate = nn.Linear(config.input_dim*2, config.input_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, memory, mask,return_fake=False):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """

        query_layer = input
        key_layer = memory
        value_layer = memory

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        attn_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))/math.sqrt(self.attention_head_size)
        mask = -1e30 * (1 - mask[:,None,None])
        attn_scores = attn_scores + mask
        attn_scores = F.softmax(attn_scores,dim=-1)
        attn_scores = self.dropout(attn_scores)

        context_layer = torch.matmul(attn_scores,value_layer).permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = torch.cat([input,context_layer],dim=-1)
        output = self.concate(output)
            
        if self.gate_method == "gate_att_up":
            gate_sg = torch.sigmoid(output)
            gate_th = torch.tanh(output)
            output = gate_sg * gate_th
        elif self.gate_method == "no_gate":
            output = F.gelu(output)
            
        return output 

class AdaptedGatedAttention(nn.Module):
    def __init__(self, config, gate_method='gate_att_up'):
        super(AdaptedGatedAttention, self).__init__()
        self.gate_method = gate_method
        if gate_method not in ['gate_att_up','no_gate']:
            raise ValueError("Not support gate method: {}".format(self.gate_method))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.input_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.input_dim, self.all_head_size)  #2.4/2.5 needs
        self.key = nn.Linear(config.input_dim, self.all_head_size)
        self.value = nn.Linear(config.input_dim, self.all_head_size)
        self.concate = nn.Linear(config.input_dim*2, config.input_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, memory, mask,return_fake=False):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """

        query_layer = self.query(input)
        key_layer = self.key(memory)
        value_layer = self.value(memory)
        # query_layer = input
        # key_layer = memory
        # value_layer = memory

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        attn_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))/math.sqrt(self.attention_head_size)
        mask = -1e30 * (1 - mask[:,None,None])
        attn_scores = attn_scores + mask
        attn_scores = F.softmax(attn_scores,dim=-1)
        attn_scores = self.dropout(attn_scores)

        context_layer = torch.matmul(attn_scores,value_layer).permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = torch.cat([input,context_layer],dim=-1)
        output = self.concate(output)
        if return_fake:
            if self.training:
                fake_attn_scores = torch.zeros_like(attn_scores).uniform_(0,1)
            else:
                fake_attn_scores = torch.ones_like(attn_scores)
            fake_attn_scores = fake_attn_scores + mask
            fake_attn_scores = F.softmax(fake_attn_scores,dim=-1)
            fake_attn_scores = self.dropout(fake_attn_scores)
            fake_output_layer = torch.matmul(fake_attn_scores,value_layer).permute(0,2,1,3).contiguous()
            fake_output_layer = fake_output_layer.view(*new_context_layer_shape)
            fake_output_layer = torch.cat([input,context_layer],dim=-1)
            fake_output_layer = self.concate(fake_output_layer)
            fake_output_layer = F.gelu(fake_output_layer)
            
        if self.gate_method == "gate_att_up":
            gate_sg = torch.sigmoid(output)
            gate_th = torch.tanh(output)
            output = gate_sg * gate_th
        elif self.gate_method == "no_gate":
            output = F.gelu(output)
            
        return output if not return_fake else (output, fake_output_layer)

class GatedAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout, gate_method='gate_att_up'):
        super(GatedAttention, self).__init__()
        self.gate_method = gate_method
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_1 = nn.Linear(memory_dim, hid_dim, bias=True)

        out_dim = hid_dim
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

class AdaptedBiAttention(nn.Module):
    def __init__(self, config):
        super(AdaptedBiAttention, self).__init__()
        if config.input_dim % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.input_dim, config.num_attention_heads)
            )
        self.output_attentions = False

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.input_dim / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.input_dim, self.all_head_size)
        self.key = nn.Linear(config.input_dim, self.all_head_size)
        self.value = nn.Linear(config.input_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None
    ):
        """
        :param input: hidden_states N * Ld * d
        :param memory: encoder_hidden_states N * Lm * d
        :param mask: attention_mask N * Lm
        :param head_mask: N * Ld
        :return:
        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_mask = -1e30 * (1 - attention_mask[:,None,None])
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            head_mask = 1 - head_mask[:,None,:,None]
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        return context_layer

class BiAttention(nn.Module):
    def __init__(self, input_dim, memory_dim, hid_dim, dropout):
        super(BiAttention, self).__init__()
        self.dropout = dropout
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(memory_dim, 1, bias=False)

        self.input_linear_2 = nn.Linear(input_dim, hid_dim, bias=True)
        self.memory_linear_2 = nn.Linear(memory_dim, hid_dim, bias=True)

        self.dot_scale = np.sqrt(input_dim)

    def forward(self, input, memory, mask, total_mask ,only_c2q=False):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :param total_mask: N * Ld
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
        att = att - 1e30 * total_mask.unsqueeze(-1)

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F.softmax(att, dim=1)         # N x M similarity matrix        #v2.1 global attn score
        weight_one = weight_one * mask.unsqueeze(1)
        output_one = torch.bmm(weight_one, memory)  #context2query   N x d
        if only_c2q:
            return output_one, memory, cross_dot
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)  # 1 x N
        output_two = torch.bmm(weight_two, input)   #query2context 1 x N N x d -> 1 x d

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1), memory, cross_dot



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


class CFPredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    """
    def __init__(self, config):
        super(CFPredictionLayer, self).__init__()
        self.config = config
        input_dim = config.input_dim

        self.start_linear = OutputLayer(input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(input_dim, config, num_answer=4)


    def forward(self, batch, context_input, cls_states=None):
        context_mask = batch['context_mask']

        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L

        if cls_states is None:
            type_prediction = self.type_linear(context_input[:, 0, :])
        else:
            type_prediction = self.type_linear(cls_states) * batch['para_mask'].unsqueeze(-1)
            type_prediction = torch.sum(type_prediction,dim=1)

        return start_prediction, end_prediction, type_prediction
