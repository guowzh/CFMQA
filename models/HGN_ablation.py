from models.layers import *
from csr_mhqa.utils import count_parameters


class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        if config.use_biattn:
            self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
            self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim) if not config.only_c2q else lambda *args,**kwargs:args[0]
        else:
            self.encode_dim2hidden_dim = nn.Linear(config.input_dim,config.hidden_dim)
        self.hidden_dim = config.hidden_dim

        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop) if config.use_lstm else lambda *args,**kwargs:args[0]
        if not self.config.use_lstm:
            self.not_lstm_linear = nn.Linear(self.hidden_dim,self.hidden_dim*2)

        self.graph_blocks = nn.ModuleList()
        for _ in range(self.config.num_gnn_layers):
            self.graph_blocks.append(GraphBlock(self.config.q_attn, config))

        if config.use_gate:
            self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim if config.q_update else config.hidden_dim*2,
                                            hid_dim=self.config.ctx_attn_hidden_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn) 
        else:
            self.ctx_linear = nn.Linear(self.hidden_dim*2,self.hidden_dim)

        q_dim = self.hidden_dim if config.q_update else config.input_dim

        self.predict_layer = PredictionLayer(self.config, q_dim)

    def forward(self, batch, return_yp):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        if self.config.use_biattn:
            attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping,self.config.only_c2q)

            input_state = self.bi_attn_linear(attn_output) # N x L x d
        else:
            input_state = self.encode_dim2hidden_dim(context_encoding)
        input_state = self.sent_lstm(input_state, batch['context_lens'])    # N x L x 2d
        if not self.config.use_lstm:
            input_state = self.not_lstm_linear(input_state)

        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        pre_input_state = input_state

        if self.config.use_gate:
            graph_state,graph_mask = self.graph_blocks[-1].get_graph_state(batch,input_state,query_vec)
            input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        else:
            input_state = self.ctx_linear(input_state)
        predictions = self.predict_layer(batch, input_state, packing_mask=query_mapping, return_yp=return_yp)

        for l in range(self.config.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = self.graph_blocks[l](batch, pre_input_state, query_vec,predictions[0]+predictions[1])

            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]
