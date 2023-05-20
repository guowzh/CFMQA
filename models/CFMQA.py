from models.cflayers import *

eps = 1e-12
# baseline + TIE(sp+ans+type) Y1+Y2+Y3-c
class CounterfactualMultihopQA(nn.Module):
    """
    CounterfactualMultihopQA
    """
    def __init__(self, config):
        super(CounterfactualMultihopQA, self).__init__()
        self.config = config

        self.para_mlp = OutputLayer(config.input_dim, config, 2)
        self.sent_mlp = OutputLayer(config.input_dim, config, 2)
        self.ent_mlp = OutputLayer(config.input_dim,config,1)
        self.ans_predict_layer = CFPredictionLayer(config)
        self.cache_S = 0
        self.cache_mask = None
        self.dropout = nn.Dropout(0.5)

        self.c_para = nn.Parameter(torch.zeros(1,2))    #random distribution assumptino
        self.c_sent = nn.Parameter(torch.zeros(1,2))
        self.c_ans_start = nn.Parameter(torch.zeros(1,512))
        self.c_ans_end = nn.Parameter(torch.zeros(1,512))
        self.c_type = nn.Parameter(torch.zeros(1,4))

        # self.c_para = nn.Parameter(torch.zeros(1))    #uniform distribution assumption
        # self.c_sent = nn.Parameter(torch.zeros(1))
        # self.c_ans_start = nn.Parameter(torch.zeros(1))
        # self.c_ans_end = nn.Parameter(torch.zeros(1))
        # self.c_type = nn.Parameter(torch.zeros(1))

    def forward(self, batch, return_yp):
        # extract query encoding
        query_mapping = batch['query_mapping']
        input_state = batch['context_encoding']
        
        # para sent ent query encode
        ent_state, sent_state, para_state = self.para_sent_ent_encode(batch,input_state)
        ent_mask = batch['ent_mask']
        ent_logits = self.ent_mlp(ent_state).squeeze(-1)
        ent_logits = ent_logits - 1e30 * (1 - ent_mask)
        para_logits = self.para_mlp(para_state)
        sent_logits = self.sent_mlp(sent_state)
        start_logits, end_logits, q_type_logits = self.ans_predict_layer(batch, input_state)

        n, max_para =  input_state.shape[1], para_state.shape[1]
        if self.training:
            contrast_state = batch['constrast_encoding']
            # contrast para ent inputs encode
            contrast_sent_state, contrast_para_state,contrast_state,cf_cls_states = self.para_sent_ent_encode(batch,contrast_state,max_para,contrast=True)
            # contrast logits 
            contrast_para_logits = self.para_mlp(contrast_para_state)
            contrast_sent_logits = self.sent_mlp(contrast_sent_state)

            perturbed_state = batch['perturbed_encoding']
            # perturbed logits
            perturbed_sent_state, perturbed_para_state, perturbed_state,pt_cls_states = self.para_sent_ent_encode(batch,perturbed_state,max_para,True)
            perturbed_para_logits = self.para_mlp(perturbed_para_state)
            perturbed_sent_logits = self.sent_mlp(perturbed_sent_state)

            cf_start,cf_end,cf_type = self.ans_predict_layer(batch,contrast_state,cls_states=cf_cls_states)
            perturbed_start,perturbed_end,perturbed_type = self.ans_predict_layer(batch,perturbed_state,cls_states=pt_cls_states)

            # Y1 + Y2 + Y3 - c
            para_logits = para_logits + contrast_para_logits + perturbed_para_logits - self.c_para
            sent_logits = sent_logits + contrast_sent_logits + perturbed_sent_logits - self.c_sent

            start_logits = start_logits + cf_start + perturbed_start - self.c_ans_start[:,:n]
            end_logits = end_logits + cf_end + perturbed_end  - self.c_ans_end[:,:n]
            q_type_logits = q_type_logits + cf_type + perturbed_type  - self.c_type

        else:
            # Y1 - c ... optional
            para_logits = para_logits - self.c_para
            sent_logits = sent_logits - self.c_sent
            start_logits = start_logits - self.c_ans_start[:,:n]
            end_logits = end_logits - self.c_ans_end[:,:n]
            q_type_logits = q_type_logits - self.c_type

        if return_yp:
            yp1, yp2 = self.get_span(start_logits,end_logits,query_mapping)
            return start_logits, end_logits, q_type_logits, para_logits, sent_logits, ent_logits, yp1, yp2
        else:
            return start_logits, end_logits, q_type_logits, para_logits, sent_logits, ent_logits,None,None

    def para_sent_ent_encode(self,batch,input_state,max_para=None,contrast=False):
        """
        input_state: bsz x n x d
        trunc_query_state: bsz x m x d
        trunc_query_mapping: bsz x m
        """
        n,d = input_state.shape[1:]
        ent_state = None
        if not contrast:
            #ent encoding
            ent_mapping = batch['ent_mapping'].permute(0,2,1).contiguous()
            ent_state = torch.matmul(ent_mapping,input_state)

            #sent_encoding
            sent_start_mapping = batch['sent_start_mapping']
            sent_state = torch.matmul(sent_start_mapping,input_state)

            #para_encoding
            para_start_mapping = batch['para_start_mapping']
            para_state = torch.matmul(para_start_mapping,input_state)
        else:
            # construct contrast inputs
            para_mapping = batch['para_mapping'].permute(0,2,1).unsqueeze(-1)   #B x max_para x n x 1
            contrast_inputs_ = input_state.view(-1,max_para,n,d)    # B x max_para x n x d
            cls_states = self.dropout(contrast_inputs_[:,:,0,:])
            contrast_inputs_ = contrast_inputs_ * para_mapping
            contrast_inputs_ = contrast_inputs_.sum(dim=1)  #B x n x d
            # sent encoding
            sent_start_mapping = batch['sent_start_mapping']    # B x max_sent x n
            sent_state = torch.matmul(sent_start_mapping,contrast_inputs_)
            #para encoding
            para_start_mapping = batch['para_start_mapping']
            para_state = torch.matmul(para_start_mapping,contrast_inputs_)

        return (ent_state, sent_state, para_state) if not contrast else (sent_state,para_state,contrast_inputs_,cls_states)

    def get_span(self,start_prediction,end_prediction,packing_mask):
        outer = start_prediction[:,:,None] + end_prediction[:,None]
        outer_mask = self.get_output_mask(outer)    #start position < end position
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return yp1, yp2

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)


