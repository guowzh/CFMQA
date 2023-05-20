from models.layers import *

# DFGN + TIE (sp+ans+type)
class GraphFusionNet(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(GraphFusionNet, self).__init__()
        self.config = config
        self.n_layers = config.num_gnn_layers
        self.max_query_length = 50

        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)

        h_dim = config.hidden_dim
        q_dim = config.hidden_dim if config.q_update else config.input_dim

        self.basicblocks = nn.ModuleList()
        self.query_update_layers = nn.ModuleList()
        self.query_update_linears = nn.ModuleList()

        for layer in range(self.n_layers):
            self.basicblocks.append(BasicBlock(h_dim, q_dim, layer, config))
            if config.q_update:
                self.query_update_layers.append(BiAttention(h_dim, h_dim, h_dim, config.bi_attn_drop))
                self.query_update_linears.append(nn.Linear(h_dim * 4, h_dim))

        q_dim = h_dim if config.q_update else config.input_dim
        if config.prediction_trans:
            self.predict_layer = TransformerPredictionLayer(self.config, q_dim)
        else:
            self.predict_layer = DFGNPredictionLayer(self.config, q_dim)
        self.dropout = nn.Dropout(0.3)

        self.c_sent = nn.Parameter(torch.zeros(1,2))
        self.c_ans_start = nn.Parameter(torch.zeros(1,512))
        self.c_ans_end = nn.Parameter(torch.zeros(1,512))
        self.c_type = nn.Parameter(torch.zeros(1,4))

    def dfgnforward(self, batch,context_encoding, return_yp=True, debug=False):
        query_mapping = batch['query_mapping']
        entity_mask = batch['ent_mask']

        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        attn_output, trunc_query_state = self.bi_attention(context_encoding, trunc_query_state, trunc_query_mapping)
        input_state = self.bi_attn_linear(attn_output)

        if self.config.q_update:
            query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        softmasks = []
        entity_state = None
        for l in range(self.n_layers):
            input_state, entity_state, softmask = self.basicblocks[l](input_state, query_vec, batch)
            softmasks.append(softmask)
            if self.config.q_update:
                query_attn_output, _ = self.query_update_layers[l](trunc_query_state, entity_state, entity_mask)
                trunc_query_state = self.query_update_linears[l](query_attn_output)
                query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)

        predictions = self.predict_layer(batch, input_state, query_vec, entity_state, query_mapping, return_yp)
        start, end, sp, Type, ent, yp1, yp2 = predictions

        if return_yp:
            return start, end, Type, None, sp,ent, yp1, yp2   #adapt to compute_loss
        else:
            return start, end, sp, Type, softmasks, ent 

    def forward(self,batch,return_yp):
        start,end,q_type,te_para,te_sent,ent,yp1,yp2 = self.dfgnforward(batch,batch['context_encoding'])
        max_para,n = batch['para_start_mapping'].shape[1:]
        if self.training:
            cf_input = self.contrast_encode(batch,batch['constrast_encoding'],max_para)
            cf_start, cf_end, cf_type,_ ,cf_sent, _,_,_ = self.dfgnforward(batch,cf_input)
            perturb_input = self.contrast_encode(batch,batch['perturbed_encoding'],max_para)
            perturb_start, perturb_end, perturb_type, _, perturb_sent, _,_,_ = self.dfgnforward(batch,perturb_input)

            te_sent = te_sent + cf_sent + perturb_sent - self.c_sent
            start = start + cf_start + perturb_start - self.c_ans_start[:,:n]
            end = end + cf_end + perturb_end - self.c_ans_end[:,:n]
            q_type = q_type + cf_type + perturb_type - self.c_type
        else:
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


