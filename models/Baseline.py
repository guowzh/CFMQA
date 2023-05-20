
from re import T
from models.layers import *

eps = 1e-12
# baseline : similar to dire
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
        self.ans_predict_layer = PredictionLayer(config,config.input_dim)
        

    def forward(self, batch, return_yp):
        # extract query encoding
        query_mapping = batch['query_mapping']
        input_state = batch['context_encoding']
        
        # para sent ent query encode
        ent_state, sent_state, para_state =   self.para_sent_ent_encode(batch,input_state)

        ent_mask = batch['ent_mask']
        ent_logits = self.ent_mlp(ent_state).squeeze(-1)
        ent_logits = ent_logits - 1e30 * (1 - ent_mask)
        para_logits = self.para_mlp(para_state)
        sent_logits = self.sent_mlp(sent_state)
        predictions = self.ans_predict_layer(batch, input_state, packing_mask=query_mapping, return_yp=return_yp)

        if return_yp:
            start_logits, end_logits, q_type_logits, yp1, yp2 = predictions
            return start_logits, end_logits, q_type_logits, para_logits, sent_logits, ent_logits, yp1, yp2
        else:
            start_logits, end_logits, q_type_logits = predictions
            return start_logits, end_logits, q_type_logits, para_logits, sent_logits, ent_logits,None,None

    def para_sent_ent_encode(self,batch,input_state):
        """
        input_state: bsz x n x d
        trunc_query_state: bsz x m x d
        trunc_query_mapping: bsz x m
        """
        #ent encoding
        ent_mapping = batch['ent_mapping'].permute(0,2,1).contiguous()
        ent_state = torch.matmul(ent_mapping,input_state)

        #sent_encoding
        sent_start_mapping = batch['sent_start_mapping']
        sent_state = torch.matmul(sent_start_mapping,input_state)

        #para_encoding
        para_start_mapping = batch['para_start_mapping']
        para_state = torch.matmul(para_start_mapping,input_state)

        return ent_state, sent_state, para_state

