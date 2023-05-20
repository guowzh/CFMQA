import envs

from transformers import (BertConfig, BertTokenizer, BertModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel,
                          AlbertConfig, AlbertTokenizer, AlbertModel,
                          XLNetConfig, XLNetTokenizer, XLNetModel)
from transformers import (BertForMaskedLM, RobertaForMaskedLM,AlbertForMaskedLM)

############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta':(RobertaConfig,RobertaModel,RobertaTokenizer),
    'albert':(RobertaConfig,AlbertModel,AlbertTokenizer),
    'xlnet' :(XLNetConfig, XLNetModel,XLNetTokenizer),
    'bertlm' :(BertConfig, BertForMaskedLM, BertTokenizer),
    'robertalm': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'albertlm': (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer),
}
