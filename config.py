import argparse
from utils import get_logger

logger = get_logger()


arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type', type=str, choices=['seq2seq', 'classification'], default='classification')
net_arg.add_argument('--nas_type', type=str, choices=['NAS_RNN'], default='NAS_RNN')
net_arg.add_argument('--nas', type=str2bool, default=True)


# Controller
net_arg.add_argument('--num_blocks', type=int, default=6)
net_arg.add_argument('--use_highway_connections', type=str2bool, default=True)
net_arg.add_argument('--tie_weights', type=str2bool, default=True)
net_arg.add_argument('--controller_hid', type=int, default=100)

# Shared parameters for PTB
net_arg.add_argument('--model_type', type=str, default='max-pool', choices=['max-pool'])
net_arg.add_argument('--dropout', type=float, default=0.5)
net_arg.add_argument('--dropoute', type=float, default=0.1)
net_arg.add_argument('--dropouti', type=float, default=0.65)
net_arg.add_argument('--dropouth', type=float, default=0.3)
net_arg.add_argument('--use_variational_dropout', type=str2bool, default=False)
net_arg.add_argument('--weight_init', type=float, default=None)
net_arg.add_argument('--cell_type', type=str, default='lstm', choices=['lstm','gru'])
net_arg.add_argument('--birnn', type=str2bool, default=True)
net_arg.add_argument('--embed', type=int, default=256) 
net_arg.add_argument('--hid', type=int, default=256)
net_arg.add_argument('--hidden_varient', type=str, default='gru', choices=['gru','simple'])
net_arg.add_argument('--use_bias', type=str2bool, default=False)
net_arg.add_argument('--num_layers', type=int, default=1)
net_arg.add_argument('--num_classes', type=int, default=2)
net_arg.add_argument('--rnn_max_length', type=int, default=35)
net_arg.add_argument('--encoder_rnn_max_length', type=int, default=50)
net_arg.add_argument('--decoder_rnn_max_length', type=int, default=20)
net_arg.add_argument('--max_vocab_size', type=int, default=10000)
net_arg.add_argument('--beam_size', type=int, default=1)
net_arg.add_argument('--rnn_activations', type=eval,
                     default="['tanh', 'ReLU', 'identity', 'sigmoid']")





# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='')
data_arg.add_argument('--vocab_file', type=str, default='data/glue_tasks/qnli/vocab')
data_arg.add_argument('--use_glove_emb', type=str2bool, default=False)
data_arg.add_argument('--use_elmo', type=str2bool, default=True)
data_arg.add_argument('--use_precomputed_elmo', type=str2bool, default=True)
data_arg.add_argument('--use_bert', type=str2bool, default=False)
data_arg.add_argument('--glove_file_path', type=str, default='')

# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test', 'retrain', 'retest', 'val'],
                       help='train: Training ENAS, derive: Deriving Architectures')
learn_arg.add_argument('--batch_size', type=int, default=64)
learn_arg.add_argument('--use_cas', type=str2bool, default=False)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--max_epoch', type=int, default=20)
learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
learn_arg.add_argument('--use_l2_regularization', type=str2bool, default=False)
learn_arg.add_argument('--l2_reg_lambda', type=float, default=1e-7)
learn_arg.add_argument('--use_block_sparse_regularization', type=str2bool, default=False)
learn_arg.add_argument('--block_sparse_reg_lambda', type=float, default=1e-7)
learn_arg.add_argument('--use_alcl_condition2', type=str2bool, default=False)
learn_arg.add_argument('--alcl_l2_reg_lambda', type=float, default=1e-7)
learn_arg.add_argument('--orthogonal_reg_lambda', type=float, default=1e-7)


# Controller
learn_arg.add_argument('--ppl_square', type=str2bool, default=False)
learn_arg.add_argument('--reward_type', type=str, default='CIDEr')
learn_arg.add_argument('--num_reward_batches', type=int, default=1)
learn_arg.add_argument('--reward_c', type=int, default=80)
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.95) 
learn_arg.add_argument('--discount', type=float, default=1.0)
learn_arg.add_argument('--controller_max_step', type=int, default=500,
                       help='step for controller parameters')
learn_arg.add_argument('--controller_optim', type=str, default='adam')
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4,
                       help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--controller_grad_clip', type=float, default=0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--entropy_coeff', type=float, default=1e-4)
learn_arg.add_argument('--use_softmax_tanh_c_temperature', type=str2bool, default=False)
learn_arg.add_argument('--use_softmax_tanh_c', type=str2bool, default=False)

# Shared parameters
learn_arg.add_argument('--initial_step', type=int, default=0)
learn_arg.add_argument('--max_step', type=int, default=200,
                       help='step for model parameters')
learn_arg.add_argument('--num_sample', type=int, default=1,
                       help='# of Monte Carlo samples')
learn_arg.add_argument('--optim', type=str, default='adam')
learn_arg.add_argument('--lr', type=float, default=0.001)
learn_arg.add_argument('--use_decay_lr', type=str2bool, default=False)
learn_arg.add_argument('--decay', type=float, default=0.96)
learn_arg.add_argument('--decay_after', type=float, default=15)
learn_arg.add_argument('--l2_reg', type=float, default=1e-7)
learn_arg.add_argument('--grad_clip', type=float, default=0.25)
learn_arg.add_argument('--use_batchnorm', type=str2bool, default=False)
learn_arg.add_argument('--use_node_batchnorm', type=str2bool, default=False)
learn_arg.add_argument('--batchnorm_momentum', type=float, default=0.1)

# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_name', type=str, default='')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--load_dag', type=str, default='')
misc_arg.add_argument('--continue_training', type=str2bool, default=False)
misc_arg.add_argument('--use_alcl', type=str2bool, default=False)
misc_arg.add_argument('--multitask', type=str, default=None)
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=1)
misc_arg.add_argument('--save_criteria', type=str, default='acc', choices=['acc','CIDEr', 'AVG', 'F1', 'invppl'])
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=1111)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)


def get_args():
    args, unparsed = parser.parse_known_args()
    #print(args.multitask)
    if args.multitask is not None:
        args.multitask = args.multitask.split(",")
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed

