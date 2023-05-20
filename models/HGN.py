from models.layers import *

class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        self.hidden_dim = config.hidden_dim

        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=1,
                                     dropout=config.lstm_drop)

        self.graph_blocks = nn.ModuleList()
        for _ in range(self.config.num_gnn_layers):
            self.graph_blocks.append(GraphBlock(self.config.q_attn, config))

        self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim if config.q_update else config.hidden_dim*2,
                                            hid_dim=self.config.ctx_attn_hidden_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)

        # q_dim = self.hidden_dim if config.q_update else config.input_dim

        self.predict_layer = PredictionLayer(self.config)
        self.dropout = nn.Dropout(0.5)

        self.c_para = nn.Parameter(torch.zeros(1,2))
        self.c_sent = nn.Parameter(torch.zeros(1,2))
        self.c_ans_start = nn.Parameter(torch.zeros(1,512))
        self.c_ans_end = nn.Parameter(torch.zeros(1,512))
        self.c_type = nn.Parameter(torch.zeros(1,4))

    def hgnforward(self, batch,context_encoding,return_yp=True):
        query_mapping = batch['query_mapping']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)

        input_state = self.bi_attn_linear(attn_output) # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])

        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        para_logits, sent_logits = [], []
        para_predictions, sent_predictions, ent_predictions = [], [], []

        for l in range(self.config.num_gnn_layers):
            new_input_state, graph_state, graph_mask, sent_state, query_vec, para_logit, para_prediction, \
            sent_logit, sent_prediction, ent_logit = self.graph_blocks[l](batch, input_state, query_vec)

            para_logits.append(para_logit)
            sent_logits.append(sent_logit)
            para_predictions.append(para_prediction)
            sent_predictions.append(sent_prediction)
            ent_predictions.append(ent_logit)

        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        predictions = self.predict_layer(batch, input_state, packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1], yp1, yp2  #adapt to compute loss
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions[-1], sent_predictions[-1], ent_predictions[-1]

    def forward(self,batch,return_yp):
        start,end,q_type,te_para,te_sent,ent,yp1,yp2 = self.hgnforward(batch,batch['context_encoding'])
        max_para,n = batch['para_start_mapping'].shape[1:]
        if self.training:
            cf_input = self.contrast_encode(batch,batch['constrast_encoding'],max_para)
            cf_start, cf_end, cf_type, cf_para, cf_sent, _,_,_ = self.hgnforward(batch,cf_input)
            perturbed_input = self.contrast_encode(batch,batch['perturbed_encoding'],max_para)
            perturbed_start, perturbed_end , perturbed_type, perturbed_para, perturbed_sent, _,_,_ = self.hgnforward(batch,perturbed_input)

            # Y1 + Y2 + Y3 - c
            te_para = te_para + cf_para + perturbed_para - self.c_para
            te_sent = te_sent + cf_sent + perturbed_sent - self.c_sent

            start = start + cf_start + perturbed_start - self.c_ans_start[:,:n]
            end = end + cf_end + perturbed_end - self.c_ans_end[:,:n]
            q_type = q_type + cf_type + perturbed_type - self.c_type

        else:
            # Y1 - c ... optional
            te_para = te_para - self.c_para
            te_sent = te_sent - self.c_sent
            start = start - self.c_ans_start[:,:n]
            end = end - self.c_ans_end[:,:n]
            q_type = q_type - self.c_type

        return start, end, q_type, te_para, te_sent, ent, yp1, yp2

    def contrast_encode(self,batch,input_state,max_para):
        n,d = input_state.shape[1:]
        # construct contrast inputs
        para_mapping = batch['para_mapping'].permute(0,2,1).unsqueeze(-1)   #B x max_para x n x 1
        query_mapping = batch['query_mapping'].repeat(1,max_para).view(-1,max_para,n).unsqueeze(-1)
        contrast_inputs_ = input_state.view(-1,max_para,n,d)    # B x max_para x n x d
        context_ = contrast_inputs_ * para_mapping
        query_ = self.dropout(contrast_inputs_ * query_mapping)
        contrast_inputs_ = (context_ + query_).sum(dim=1)
        return contrast_inputs_
