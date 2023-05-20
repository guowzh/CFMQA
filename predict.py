from operator import ge
from collections import OrderedDict
import numpy as np
import logging
import sys

from os.path import join

from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from csr_mhqa.data_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import *

from models.CFMQA import *
from model_envs import MODEL_CLASSES, ALL_MODELS
from models.DFGN import *
from models.HGN import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
args = complete_default_train_parser(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Read Data
##########################################################################
helper = DataHelper(gz=True, config=args)

# Set datasets
dev_example_dict,dev_feature_dict,dev_dataloader = [],[],[]
dev_example_dict.append(helper.dev_dire_example_dict)
dev_feature_dict.append(helper.dev_dire_feature_dict)
dev_dataloader.append(helper.dev_dire_loader)
dev_example_dict.append(helper.probe_dev_example_dict)
dev_feature_dict.append(helper.probe_dev_feature_dict)
dev_dataloader.append(helper.probe_dev_loader)

#########################################################################
# Initialize Model
##########################################################################
config_class, model_encoder, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.encoder_name_or_path)

encoder_path = join(args.exp_name, 'encoder_{}.pkl'.format(args.ckpt_num))
model_path = join(args.exp_name, 'model_{}.pkl'.format(args.ckpt_num))
logger.info("Loading encoder from: {}".format(encoder_path))
logger.info("Loading model from: {}".format(model_path))

encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
model = CounterfactualMultihopQA(config=args)
# model = GraphFusionNet(config=args)
# model = HierarchicalGraphNetwork(config=args)

def get_new_state_dict(state_dict):
    # if model save in DataParallel
    if list(state_dict.keys())[0][:7] != "module.":
        return state_dict
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

if encoder_path is not None:
    encoder_state_dict = torch.load(encoder_path)
    encoder_state_dict = get_new_state_dict(encoder_state_dict)
    encoder.load_state_dict(encoder_state_dict)
if model_path is not None:
    model_state_dict = torch.load(model_path)
    model_state_dict = get_new_state_dict(model_state_dict)
    model.load_state_dict(model_state_dict)

encoder.to(args.device)
model.to(args.device)

encoder.eval()
model.eval()

#########################################################################
# Evaluation
##########################################################################
from envs import DATASET_FOLDER

output_pred_file = [join(args.exp_name, 'original_pred.json'), join(args.exp_name,'probe_pred.json')]
output_eval_file = [join(args.exp_name, 'original_eval.txt'), join(args.exp_name, "probe_eval.txt")]
dire_output_pred_file = [join(args.exp_name,"original_pred_dire.jsonl"),\
                        join(args.exp_name,"probe_pred_dire.jsonl")]
for i in range(len(dev_example_dict)):
    metrics, threshold = eval_model(args, encoder, model,
                                    dev_dataloader[i], dev_example_dict[i], dev_feature_dict[i],
                                    output_pred_file[i], output_eval_file[i], args.dev_gold_file)
    print("convert to dire evaluate jsonl.")
    convert_hgn_to_dire_pred(output_pred_file[i],dire_output_pred_file[i])
    # print("Best threshold: {}".format(threshold))
    # for key, val in metrics.items():
    #     print("{} = {}".format(key, val))

original_dev_file = join(DATASET_FOLDER,"data_raw/original_hotpotqa_dev.json")
probe_of_original_dev_file = join(DATASET_FOLDER,"data_raw/probe_of_original_hotpotqa_dev.json")
comparison_eval_output = join(args.exp_name,"dire_eval_ckpt={}.txt".format(args.ckpt_num))
cmd = "cd dire_evaluate;bash evaluate.sh {} {} {} {} {}".format(original_dev_file,probe_of_original_dev_file,\
    dire_output_pred_file[0].replace(" ","\ "),dire_output_pred_file[1].replace(" ","\ "),comparison_eval_output.replace(" ", "\ "))
print("start...")
print(cmd)
os.system(cmd)
with open(comparison_eval_output,"r") as fin:
    print(fin.read())