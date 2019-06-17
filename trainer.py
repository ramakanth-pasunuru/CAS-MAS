from __future__ import print_function

import math
import sys
import scipy.signal
from glob import glob
import json
import copy
from collections import defaultdict, deque
import ipdb
from sklearn.metrics import f1_score

import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.parallel
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable

from allennlp.modules.elmo import Elmo, batch_to_ids

from utils import *
from models import *
from tensorboard import TensorBoard
from common_loader import *



# requirements for ELMo
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


logger = get_logger()


controller_type = {"NAS_RNN": Controller}


def discount(x, discount):
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

def to_var(args, x, volatile=False):
    if args.cuda:
        x = x.cuda()
    return Variable(x, volatile=volatile)

def get_list_parameters(model):
    if isinstance(model, list):
        params = []
        for m in model:
            params.extend(m.parameters())
    else:
        params = model.parameters()

    return params


def get_optimizer(name):
    if name.lower() == "sgd":
        optim = t.optim.SGD
    elif name.lower() == "adam":
        optim = t.optim.Adam
    elif name.lower() == 'rmsprop':
        optim = t.optim.RMSprop

    return optim



class Trainer(object):
    def __init__(self, args, dataset):
        self.args = args
        self.cuda = args.cuda
        self.dataset = dataset
        if args.network_type in ['seq2seq', 'classification'] and \
        args.dataset in ['msrvtt', 'msvd', 'didemo', 'multitask', 'fnc', \
                        'qnli', 'wnli', 'rte', 'multitask_glue']:
            self.train_data = dataset['train']
            self.valid_data = dataset['val']
            self.test_data = dataset['test']
        else:
            raise Exception(f"Unknown network type: {args.network_type} and unknown dataset: {args.dataset} combination !!")

        if args.use_tensorboard and self.args.mode == 'train':
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        self.build_model()


        if self.args.load_path:
            self.load_model()

        

    def build_model(self):
        self.start_epoch = self.epoch = 0
        self.step = 0

        if self.args.network_type == 'classification':
            if self.args.model_type in ['max-pool']:
                self.model = RTEClassifier(self.args)
            else:
                raise Exception(f"unknown model type: {self.args.model_type}")
                
            if self.args.use_glove_emb and self.args.mode == 'train':
                logger.info('Loading Glove Embeddings...')
                embeddings = load_glove_emb(self.args.glove_file_path, self.train_data.vocab)
                self.model.embed.weight.data = embeddings
        else:
            raise NotImplemented(f"Network type `{self.args.network_type}` is not defined")

        if self.args.use_elmo and not self.args.use_precomputed_elmo:
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            self.elmo.cuda()

        if self.args.num_gpu == 1:
            self.model.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplemented("`num_gpu > 1` is in progress")

        self.ce = nn.CrossEntropyLoss()
        logger.info(f"[*] # Parameters: {self.count_parameters}")

    def train(self):
        # add file loging handder for logger
        hdlr = logging.FileHandler(os.path.join(self.args.model_dir,'training.log'))
        logger.addHandler(hdlr)
        optimizer = get_optimizer(self.args.optim)
        self.optim = optimizer(
                self.model.parameters(),
                lr=self.args.lr)

        for self.epoch in range(self.start_epoch, self.args.max_epoch):  
            if self.args.network_type == 'rnn':
                self.train_lm()
            else:
                self.train_model()
            if self.epoch % self.args.save_epoch == 0:
                if self.args.network_type == 'rnn':
                    scores = self.test_lm(mode='val')
                else:
                    scores = self.test(mode='val')
                self.save_model(save_criteria_score=scores)

            if self.epoch >= self.args.decay_after:
                if self.args.network_type in ['classification', 'seq2seq'] and self.args.use_decay_lr:
                    update_lr(self.optim, self.model_lr)

    def train_model(self):
        total_loss = 0
        model = self.model
        model.train()
 
        pbar = tqdm(total=self.train_data.num_steps, desc="train_model")

        batcher = self.train_data.get_batcher()

        for step in range(0,self.train_data.num_steps): 
            batch = next(batcher)
            if self.args.network_type == 'classification' and self.args.dataset in ['multitask','qnli', 'rte', 'wnli', 'multitask_glue']:
                if self.args.use_elmo and not self.args.use_precomputed_elmo:
                    seq1 = batch.get('original_seq1')
                    seq1_list = []
                    for s in seq1:
                        seq1_list.append(s.split(' '))
                    seq1 = self.elmo(batch_to_ids(seq1_list).cuda())
                    seq1_len = seq1['mask'].sum(1)
                    seq1 = seq1['elmo_representations'][0]

                    seq2 = batch.get('original_seq2')
                    seq2_list = []
                    for s in seq2:
                        seq2_list.append(s.split(' '))

                    seq2 = self.elmo(batch_to_ids(seq2_list).cuda())
                    seq2_len = seq2['mask'].sum(1)
                    seq2 = seq2['elmo_representations'][0]
                    
                else:
               	    seq1 = batch.get('seq1_batch')
                    seq1_len = batch.get('seq1_length')
                    seq1 = to_var(self.args, seq1)
                    seq2 = batch.get('seq2_batch')
                    seq2_len = batch.get('seq2_length')
                    seq2 = to_var(self.args, seq2)
 
                
                label = batch.get('label')
                label = to_var(self.args, label)
                logits, probs, preds = self.model(seq1, seq1_len, seq2, seq2_len)
                loss = self.ce(logits, label)

            else:
                raise Exception(f"Unknown network type: {self.args.network_type}")
            # update
            self.optim.zero_grad()
            
            loss.backward()

            t.nn.utils.clip_grad_norm(
                    model.parameters(), self.args.grad_clip)
            self.optim.step()

            total_loss += loss.data
            pbar.set_description(f"train_model| loss: {loss.data[0]:5.3f}")
           
            #break

            if step % self.args.log_step == 0 and step > 0:
                cur_loss = total_loss[0] / self.args.log_step
                ppl = math.exp(cur_loss)

                logger.info(f'| epoch {self.epoch:3d} | lr {self.args.lr:8.6f} '
                            f'| loss {cur_loss:.2f} | ppl {ppl:8.2f}')

                # Tensorboard
                if self.tb is not None:
                    self.tb.scalar_summary("model/loss", cur_loss, self.step)
                    self.tb.scalar_summary("model/perplexity", ppl, self.step)

                total_loss = 0

            step += 1
            self.step += 1
            pbar.update(1) 


    def test(self, mode):

        self.model.eval()
        counter = 0
        if mode == 'val':
            batcher = self.valid_data.get_batcher()
            num_steps = self.valid_data.num_steps
        elif mode == 'test':
            if self.test_data is None: # for GLUE tasks, there is no test set
                batcher = self.valid_data.get_batcher()
                num_steps = self.valid_data.num_steps
            else:
                batcher = self.test_data.get_batcher()
                num_steps = self.test_data.num_steps
        else:
            raise Exception("Unknow mode: {}".format(mode))

        if self.args.network_type == 'classification' and self.args.dataset in ['multitask', 'qnli', 'rte', 'wnli', 'multitask_glue']:
            true_label = []
            pred_label = []
            for i in range(num_steps):
                batch = next(batcher)
                if self.args.use_elmo and not self.args.use_precomputed_elmo:
                    seq1 = batch.get('original_seq1')
                    seq1_list = []
                    for s in seq1:
                        seq1_list.append(s.split(' '))
                    seq1 = self.elmo(batch_to_ids(seq1_list).cuda())
                    seq1_len = seq1['mask'].sum(1)
                    seq1 = seq1['elmo_representations'][0]
                    seq2 = batch.get('original_seq2')
                    seq2_list = []
                    for s in seq2:
                        seq2_list.append(s.split(' '))
                    seq2 = self.elmo(batch_to_ids(seq2_list).cuda())
                    seq2_len = seq2['mask'].sum(1)
                    seq2 = seq2['elmo_representations'][0]

               	else:
                    seq1 = batch.get('seq1_batch')
                    seq1_len = batch.get('seq1_length')
                    seq1 = to_var(self.args, seq1)
                    seq2 = batch.get('seq2_batch')
                    seq2_len = batch.get('seq2_length')
                    seq2 = to_var(self.args, seq2)
 

                label = batch.get('label')
                label = to_var(self.args, label)
                logits, probs, preds = self.model(seq1, seq1_len, seq2, seq2_len)
                true_label.extend(label.cpu().data.numpy().tolist())
                pred_label.extend(preds.cpu().data.numpy().tolist())
                #ipdb.set_trace()
            # order of labels is AGAINST, NONE, FAVOUR
            if self.args.dataset in ['stance', 'target', 'multitask']:
                f1scores = f1_score(true_label, pred_label, average=None)
                f1score = np.mean([f1scores[0], f1scores[2]])
                if mode == 'val':
                    self.tb.scalar_summary(f"test/{mode}_F1", f1score, self.epoch)
                logger.info(f"{mode} F1 score: {f1score}")
            elif self.args.dataset in ['absa_laptop', 'absa_restaurant', 'fnc', 'qnli', 'wnli', 'rte', 'multitask_glue']:
                if mode == 'test' and self.args.dataset in ['qnli', 'rte', 'wnli']: # glue hidden test set evaluation
                    if not os.path.exists(os.path.join(self.args.model_dir,'results')):
                        os.mkdir(os.path.join(self.args.model_dir,'results'))
                    
                    file_loc = self.create_result_path(filename=f'test_{self.args.dataset}.tsv')
                    logger.info(f'Writing the test results to {file_loc}')
                    fw = open(file_loc,'w')
                    fw.write('index\tprediction\n')
                    for ind, l in enumerate(pred_label):
                        if l == 0:
                            if self.args.dataset in ['wnli']:
                                fw.write(str(ind)+'\t'+'0\n')
                            elif self.args.dataset in ['qnli','rte']:
                                fw.write(str(ind)+'\t'+'entailment\n')
                        else:
                            if self.args.dataset in ['wnli']:
                                fw.write(str(ind)+'\t'+'1\n')
                            elif self.args.dataset in ['qnli', 'rte']:
                                fw.write(str(ind)+'\t'+'not_entailment\n')

                        fw.flush()

                    fw.close()


                else:
                    acc = 100.0 * (np.array(true_label)==np.array(pred_label)).sum()
                   
                    acc = acc/(1.0*len(true_label))
                    if mode == 'val' and self.tb is not None:
                        self.tb.scalar_summary(f"test/{mode}_accuracy", acc, self.epoch)
                    logger.info(f"{mode} accuracy: {acc}")

            save_criteria_score = None

            if not (mode == 'test' and self.args.dataset in ['qnli', 'rte', 'wnli']):
                
                if self.args.save_criteria == 'F1':
                    save_criteria_score = f1score
                elif self.args.save_criteria == 'acc':
                    save_criteria_score = acc


            if mode == 'test' and self.test_data is None:
                # save the result
                if not os.path.exists(os.path.join(self.args.model_dir,'results')):
                    os.mkdir(os.path.join(self.args.model_dir,'results'))

                result_save_path = self.result_path
                final_dict = {}
                final_dict['args'] = self.args.__dict__
                if self.args.dataset in ['multitask']:
                    final_dict['scores'] = f1score
                elif self.args.dataset in ['qnli', 'wnli', 'rte', 'multitask_glue']:
                    final_dict['scores'] = acc
                with open(result_save_path, 'w') as fp:
                    json.dump(final_dict, fp, indent=4, sort_keys=True)
                file_loc = self.create_result_path(filename='true_labels.txt')
                print(file_loc)
                with open(file_loc, 'w') as f:
                    f.write('\n'.join([str(l) for l in true_label]))
                file_loc = self.create_result_path(filename='pred_labels.txt')
                with open(file_loc, 'w') as f:
                    f.write('\n'.join([str(l) for l in pred_label]))
                
                file_loc = self.create_result_path(filename='correct_scores.txt')
                with open(file_loc, 'w') as f:
                    f.write('\n'.join([str(1.0) if l else str(0.0) for l in (np.array(true_label)==np.array(pred_label)).tolist()]))


            
            return save_criteria_score


    
    def save_model(self, save_criteria_score=None):
        t.save(self.model.state_dict(), self.model_path)
        logger.info(f"[*] SAVED: {self.model_path}")
        epochs, steps  = self.get_saved_models_info()
        if save_criteria_score is not None:
            if os.path.exists(os.path.join(self.args.model_dir,'checkpoint_tracker.dat')):
                checkpoint_tracker = t.load(os.path.join(self.args.model_dir,'checkpoint_tracker.dat'))
            else:
                checkpoint_tracker = {}
            key = f"{self.epoch}_{self.step}"
            value = save_criteria_score
            checkpoint_tracker[key] = value
            if len(epochs)>=self.args.max_save_num:
                low_value = 100000.0
                remove_key = None
                for key,value in checkpoint_tracker.items():
                    if low_value > value:
                        remove_key = key
                        low_value = value
                del checkpoint_tracker[remove_key]
                remove_epoch = remove_key.split("_")[0]
                paths = glob(os.path.join(self.args.model_dir,f'*_epoch{remove_epoch}_*.pth'))
                for path in paths:
                    remove_file(path)

            # save back the checkpointer tracker
            t.save(checkpoint_tracker, os.path.join(self.args.model_dir,'checkpoint_tracker.dat'))
        else:
            for epoch in epochs[:-self.args.max_save_num]:
                paths = glob(os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))
                for path in paths:
                    remove_file(path)


    def get_saved_models_info(self):
        paths = glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        steps = get_numbers(basenames, '_', 2, 'step')

        epochs.sort()
        steps.sort()
        return epochs, steps
    
    def load_model(self):

        
        if self.args.load_path.endswith('.pth'):
            map_location = None
            s = self.args.load_path.split('/')[-1]
            s = s[:-4].split('_')
            self.epoch = int(s[1][5:])
            self.step = int(s[2][4:])
            self.model.load_state_dict(
                    t.load(self.load_path, map_location=map_location))
            logger.info(f"[*] LOADED: {self.load_path}")

        else:
            if os.path.exists(os.path.join(self.args.load_path,'checkpoint_tracker.dat')):
                checkpoint_tracker = t.load(os.path.join(self.args.load_path,'checkpoint_tracker.dat'))
                best_key = None
                best_score = -1.0
                for key,value in checkpoint_tracker.items():
                    if value>best_score:
                        best_score = value
                        best_key = key


                self.epoch = int(best_key.split("_")[0])
                self.step = int(best_key.split("_")[1])

            else:
                epochs, steps = self.get_saved_models_info()

                if len(epochs) == 0:
                    logger.info(f"[!] No checkpoint found in {self.args.model_dir}...")
                    return

                self.epoch = self.start_epoch = max(epochs)
                self.step = max(steps)
            
            if self.args.num_gpu == 0:
                map_location = lambda storage, loc: storage
            else:
                map_location = None

            self.model.load_state_dict(
                    t.load(self.load_path, map_location=map_location))
            logger.info(f"[*] LOADED: {self.load_path}")

    def get_batch(self, source, idx, length=None, evaluation=False):
        # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        length = min(length if length else self.max_length, len(source) - 1 - idx)
        data = Variable(source[idx:idx+length], volatile=evaluation)
        target = Variable(source[idx+1:idx+1+length].view(-1))
        return data, target


    def create_result_path(self, filename):
        return f'{self.args.model_dir}/results/model_epoch{self.epoch}_step{self.step}_{filename}'


    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def model_path(self):
        return f'{self.args.model_dir}/model_epoch{self.epoch}_step{self.step}.pth'

    @property
    def load_path(self):
        if self.args.load_path.endswith('.pth'):
            return f'{self.args.model_dir}/model_epoch{self.epoch}_step{self.step}.pth'

        else:
            return f'{self.args.load_path}/model_epoch{self.epoch}_step{self.step}.pth'

    @property
    def result_path(self):
        return f'{self.args.model_dir}/results/model_epoch{self.epoch}_step{self.step}.json'

    @property
    def output_result_path(self):
        return f'{self.args.model_dir}/results/output_model_epoch{self.epoch}_step{self.step}.txt'

    @property
    def model_lr(self):
        degree = max(self.epoch - self.args.decay_after + 1, 0)
        return self.args.lr * (self.args.decay ** degree)



class TrainerNAS(object):
    def __init__(self, args, dataset):
        self.args = args
        self.cuda = args.cuda
        self.dataset = dataset
        if args.network_type in ['seq2seq','classification'] and \
                args.dataset in ['msrvtt', 'msvd', 'didemo', 'multitask', \
                'qnli', 'rte', 'wnli', 'multitask_glue', 'multitask_vc']:
            self.train_data = dataset['train']
            self.valid_data = dataset['val']
            self.test_data = dataset['test']
        else:
            raise Exception(f"Unknown network type: args.network_type and unknown dataset: args.dataset combination !!")

        
        self.max_length = self.args.rnn_max_length

        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        self.build_model()
        
        if self.args.load_path and self.args.mode != 'retrain': # just load the graph of DAG later
            self.load_model(self.args.load_path)


    def build_model(self):
        self.start_epoch = self.epoch = 0
        self.shared_step, self.controller_step = 0, 0

        if self.args.network_type == 'classification':
            self.shared = RTEClassifier(self.args)
            if self.args.use_glove_emb and self.args.mode in ['train','retrain']:
                logger.info('Loading Glove Embeddings...')
                if self.args.multitask is None:
                    embeddings = load_glove_emb(self.args.glove_file_path, self.train_data.vocab)
                else:
                    embeddings = load_glove_emb(self.args.glove_file_path, self.train_data['qnli'].vocab)
                self.shared.embed.weight.data = embeddings 
        else:
            raise NotImplemented(f"Network type `{self.args.network_type}` is not defined")

        self.controller = controller_type[self.args.nas_type](self.args)
  
        if self.args.use_elmo and not self.args.use_precomputed_elmo:
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            self.elmo.cuda()
        
        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplemented("`num_gpu > 1` is in progress")

        self.ce = nn.CrossEntropyLoss()

        logger.info(f"[*] # Parameters: {self.count_parameters}")

    def train(self):
        # add file loging handder for logger
        hdlr = logging.FileHandler(os.path.join(self.args.model_dir,'training.log'))
        logger.addHandler(hdlr)

        if self.args.use_alcl:
            self.dag_ref = self.get_best_dag(os.path.join(self.args.load_path,'best_dag.dat'))
            self.shared_ref = copy.deepcopy(self.shared)
            self.remove_gradient_update(self.shared_ref)

        # add file loging handder for logger
        hdlr = logging.FileHandler(os.path.join(self.args.model_dir,'training.log'))
        logger.addHandler(hdlr)

        shared_optimizer = get_optimizer(self.args.optim)
        controller_optimizer = get_optimizer(self.args.controller_optim)

        self.shared_optim = shared_optimizer(
                self.shared.parameters(),
                lr=self.shared_lr)

        self.controller_optim = controller_optimizer(
                get_list_parameters(self.controller),
                lr=self.args.controller_lr)

        if self.args.initial_step > 0:
            self.train_shared_seq2seq(max_step=self.args.initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters ω of the child models
            self.train_shared()

            # 2. Training the controller parameters θ
            self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                if self.epoch >= 0:
                    best_dag = self.derive()
                    if self.args.multitask is None:
                        scores = self.test_model(best_dag, "val")

                self.save_model()

            if self.epoch >= self.args.decay_after:
                if self.args.network_type in ['classification', 'seq2seq'] and self.args.use_shared_decay_lr:
                    update_lr(self.shared_optim, self.shared_lr)



    def retrain(self):

        # change the model directory
        # add file loging handder for logger
        hdlr = logging.FileHandler(os.path.join(self.args.model_dir,'training.log'))
        logger.addHandler(hdlr)

        """ deprecating it and using the new way of finding the best dag from networks saved"""
        #best_dag = self.derive()
        best_dag = self.get_best_dag()

        # save this dag to a file
        logger.info(f"Saving the best dag as best_dag.dat")
        t.save(best_dag, os.path.join(self.args.model_dir,'best_dag.dat'))
        best_dag = [best_dag]
        # build the model again to retrain from scratch
        self.build_model()

        if self.args.load_path and self.args.continue_training:
            self.load_model(self.args.load_path)

        if self.args.use_alcl:
            self.dag_ref = self.get_best_dag(os.path.join(self.args.load_path,'best_dag.dat'))
            self.load_model(self.args.load_path)
            self.epoch = 0
            self.step = 0
            self.shared_ref = copy.deepcopy(self.shared)
            self.remove_gradient_update(self.shared_ref)

        shared_optimizer = get_optimizer(self.args.optim)

        self.shared_optim = shared_optimizer(
                self.shared.parameters(),
                lr=self.shared_lr)

        prev_score_lr_track = None
        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters ω of the child models
            self.train_shared(best_dag=best_dag)

            if self.epoch % self.args.save_epoch == 0:
                if self.epoch >= 0:
                    if self.args.network_type in ['rnn', 'cnn']:
                        
                        save_criteria_score = self.test_lm(best_dag[0], 'val')
                        if prev_score_lr_track is None:
                            prev_score_lr_track = save_criteria_score
                            self.decay_counter = 0
                        else:
                            if save_criteria_score < prev_score_lr_track and self.args.optim == 'adam' and self.args.use_shared_decay_lr:
                                self.decay_counter += 1
                                update_lr(self.shared_optim, self.args.lr*(self.args.decay**self.decay_counter))

                            prev_score_lr_track = save_criteria_score

                    else:
                        
                        save_criteria_score = self.test_model(best_dag[0], "val")

                self.save_model(save_criteria_score=save_criteria_score)

            if self.epoch >= self.args.decay_after:
                if self.args.network_type in ['classification','seq2seq'] and self.args.use_shared_decay_lr:
                    update_lr(self.shared_optim, self.shared_lr)

    


    def train_shared(self, best_dag=None, max_step=None):
        total_loss = 0

        model = self.shared
        model.train()
 
        if self.args.multitask is None:
            batcher = self.train_data.get_batcher()
        else:
            batcher = {}
            for task in self.args.multitask:
                batcher[task] = self.train_data[task].get_batcher()

        if max_step is None:
            if self.args.mode=='retrain':
                max_step = self.train_data.num_steps
            else: 
                max_step = self.args.max_step
        else:
            if self.args.multitask is None:
                max_step = min(self.train_data.num_steps, max_step)
            else:
                min_task_step = None
                for k,v in self.train_data.items():
                    if min_task_step is None:
                        min_task_step = v.num_steps
                    else:
                        if v.num_steps < min_task_step:
                            min_task_step = v.num_steps

                max_step = min(min_task_step, max_step)

        pbar = tqdm(total=max_step, desc="train_shared")
        for step in range(0,max_step): 

            if best_dag is None:
                if isinstance(self.controller,list):
                    dags = []
                    for contr in self.controller:
                        d = contr.sample(self.args.num_sample)
                        dags.append(d[0])
                    dags = [dags]

                else:
                    dags = self.controller.sample(self.args.num_sample)
            else:
                dags = best_dag
            if self.args.multitask is None:
                batch = next(batcher)
            else:
                current_task = self.args.multitask[step%len(self.args.multitask)]
                batch = next(batcher[current_task])
                

            if self.args.network_type == 'classification' and self.args.dataset in ['multitask', 'qnli', 'rte', 'wnli', 'multitask_glue']:
                if self.args.use_elmo and not self.args.use_precomputed_elmo:
                    seq1 = batch.get('original_seq1')
                    seq1_list = []
                    for s in seq1:
                        seq1_list.append(s.split(' '))
                    seq1 = self.elmo(batch_to_ids(seq1_list).cuda())
                    seq1_len = seq1['mask'].sum(1)
                    seq1 = seq1['elmo_representations'][0]
                    seq2 = batch.get('original_seq2')
                    seq2_list = []
                    for s in seq2:
                        seq2_list.append(s.split(' '))
                    seq2 = self.elmo(batch_to_ids(seq2_list).cuda())
                    seq2_len = seq2['mask'].sum(1)
                    seq2 = seq2['elmo_representations'][0]
                else:
                    seq1 = batch.get('seq1_batch')
                    seq1_len = batch.get('seq1_length')
                    seq1 = to_var(self.args, seq1)
                    seq2 = batch.get('seq2_batch')
                    seq2_len = batch.get('seq2_length')
                    seq2 = to_var(self.args, seq2)
                label = batch.get('label')
                label = to_var(self.args, label)
                logits, probs, preds = self.shared.forward_nas(seq1, seq1_len, seq2, seq2_len, dags[0])
                if self.args.multitask is not None:
                    logits = logits[current_task]
                loss = self.ce(logits, label)

            # update
            self.shared_optim.zero_grad()
            if self.args.use_l2_regularization:
                reg = self.get_l2_regularization()
                #ipdb.set_trace()
                loss += reg
            if self.args.use_block_sparse_regularization:
                reg = self.get_block_sparse_regularization()
                loss += reg
            if self.args.use_alcl_condition2:
                reg = self.get_condition2_regularization()
                loss += reg

            if np.isnan(loss.cpu().data.numpy()): #or loss.cpu().data.numpy()>20:
                break

            loss.backward()
            #ipdb.set_trace()
            t.nn.utils.clip_grad_norm(
                    model.parameters(), self.args.grad_clip)
            self.shared_optim.step()

            #ipdb.set_trace()

            total_loss += loss.data
            pbar.set_description(f"train_shared| loss: {loss.data[0]:5.3f}")

            if step % self.args.log_step == 0 and step > 0:
                cur_loss = total_loss[0] / self.args.log_step
                if not self.args.network_type in ['classification','seq2seq']:
                    ppl = math.exp(cur_loss)
                else:
                    ppl = 0 # ignoring ppl for classification

                logger.info(f'| epoch {self.epoch:3d} | lr {self.shared_lr:4.2f} '
                            f'| loss {cur_loss:.2f} | ppl {ppl:8.2f}')

                # Tensorboard
                if self.tb is not None:
                    self.tb.scalar_summary("shared/loss", cur_loss, self.shared_step)
                    self.tb.scalar_summary("shared/perplexity", ppl, self.shared_step)

                total_loss = 0

            step += 1
            self.shared_step += 1

            pbar.update(1)


    def get_reward(self, dag, entropies):

        if type(entropies) is not np.ndarray:
            if not isinstance(entropies, list) and entropies is not None:
                entropies = entropies.data.cpu().numpy()
        if self.args.multitask is None:
            batcher = self.valid_data.get_batcher()
        else:
            batcher = {}
            for task in self.args.multitask:
                batcher[task] = self.valid_data[task].get_batcher()

        rewards = 0.0
        if self.args.multitask is not None:
            self.args.num_reward_batches = len(self.args.multitask)

        for ind in range(self.args.num_reward_batches):
            if self.args.multitask is None:
                batch = next(batcher)
            else:
                batch = next(batcher[self.args.multitask[ind]])
            
            if self.args.network_type == 'classification' and self.args.dataset in ['multitask', 'qnli', 'rte', 'wnli', 'multitask_glue']:
                if self.args.use_elmo and not self.args.use_precomputed_elmo:
                    seq1 = batch.get('original_seq1')
                    seq1_list = []
                    for s in seq1:
                        seq1_list.append(s.split(' '))
                    seq1 = self.elmo(batch_to_ids(seq1_list).cuda())
                    seq1_len = seq1['mask'].sum(1)
                    seq1 = seq1['elmo_representations'][0]

                    seq2 = batch.get('original_seq2')
                    seq2_list = []
                    for s in seq2:
                        seq2_list.append(s.split(' '))

                    seq2 = self.elmo(batch_to_ids(seq2_list).cuda())
                    seq2_len = seq2['mask'].sum(1)
                    seq2 = seq2['elmo_representations'][0]
                else:
                    seq1 = batch.get('seq1_batch')
                    seq1_len = batch.get('seq1_length')
                    seq1 = to_var(self.args, seq1)
                    seq2 = batch.get('seq2_batch')
                    seq2_len = batch.get('seq2_length')
                    seq2 = to_var(self.args, seq2)
                label = batch.get('label')
                label = to_var(self.args, label)
                logits, probs, preds = self.shared.forward_nas(seq1, seq1_len, seq2, seq2_len, dag[0])
                if self.args.multitask is not None:
                    preds = preds[self.args.multitask[ind]]

                true_label = label.cpu().data.numpy().tolist()
                pred_label = preds.cpu().data.numpy().tolist()

                # order of labels is AGAINST, NONE, FAVOUR
                if self.args.dataset in ['stance', 'target', 'multitask']:
                    f1scores = f1_score(true_label, pred_label, average=None)
                    f1score = np.mean([f1scores[0], f1scores[2]])
                    rewards += f1score
                elif self.args.dataset in ['absa_laptop', 'absa_restaurant', 'fnc', 'qnli', 'rte', 'wnli', 'multitask_glue']:
                    acc = (np.array(true_label)==np.array(pred_label)).sum()
                    acc = (1.0*acc)/len(true_label)
                    rewards += acc

        rewards = rewards/self.args.num_reward_batches

        
        return np.array([rewards])


    
    def train_controller(self):
        total_loss = 0

        model = self.controller
        if isinstance(model, list):
            for m in model:
                m.train()
        else:
            model.train()

        pbar = trange(self.args.controller_max_step, desc="train_controller")

        baseline, avg_reward_base = None, None
        reward_history, adv_history, entropy_history = [], [], []

        valid_idx = 0

        for step in pbar:
            # sample models
            dags, log_probs, entropies = self.controller.sample(with_details=True)
            # calculate reward
            if not isinstance(entropies, list):
                np_entropies = entropies.data.cpu().numpy()
            else:
                np_entropies = None
                for ent in entropies:
                    if np_entropies is None:
                        np_entropies = ent.data.cpu().numpy()
                    else:
                        np_entropies = np_entropies + ent.data.cpu().numpy()
            if self.args.network_type in ['seq2seq', 'classification']:
                rewards = self.get_reward(dags, np_entropies)
            else:
                rewards = self.get_reward(dags, np_entropies, valid_idx)

            #logger.info(f'{dags}') 


            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if self.args.use_scst:
                baseline = self.get_reward([lstm_dag()], entropies=None, valid_idx=0)
            else:
                if baseline is None:
                    baseline = rewards
                else:
                    decay = self.args.ema_baseline_decay
                    baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = - log_probs * get_variable(adv, self.cuda, requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum() # or loss.mean()
            pbar.set_description(
                    f"train_controller| R: {rewards.mean():8.6f} | R-b: {adv.mean():8.6f} "
                    f"| loss: {loss.cpu().data[0]:8.6f}")

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                t.nn.utils.clip_grad_norm(
                        model.parameters(), self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += loss.data[0]

            if step % self.args.log_step == 0 and step > 0:
                cur_loss = total_loss / self.args.log_step

                avg_reward = np.mean(reward_history)
                avg_entropy = np.mean(entropy_history)
                avg_adv = np.mean(adv_history)

                if avg_reward_base is None:
                    avg_reward_base = avg_reward

                logger.info(f'| epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
                            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
                            f'| loss {cur_loss:.5f}')

                # Tensorboard
                if self.tb is not None:
                    self.tb.scalar_summary("controller/loss", cur_loss, self.controller_step)
                    self.tb.scalar_summary(
                            "controller/reward", avg_reward, self.controller_step)
                    self.tb.scalar_summary(
                            "controller/reward-B_per_epoch", avg_reward - avg_reward_base, self.controller_step)
                    self.tb.scalar_summary(
                            "controller/entropy", avg_entropy, self.controller_step)
                    self.tb.scalar_summary(
                            "controller/adv", avg_adv, self.controller_step)

                    paths = []
                    for dag in dags:
                        if isinstance(dag, list):
                            for ind,d in enumerate(dag):
                                fname = f"{self.epoch:03d}-{self.controller_step:06d}-{avg_reward:6.4f}-{ind}.png"
                                path = os.path.join(self.args.model_dir, "networks", fname)
                                draw_network(d, path)
                                paths.append(path)

                        else:
                            fname = f"{self.epoch:03d}-{self.controller_step:06d}-{avg_reward:6.4f}.png"
                            path = os.path.join(self.args.model_dir, "networks", fname)
                            draw_network(dag, path)
                            paths.append(path)

                    self.tb.image_summary("controller/sample", paths, self.controller_step)

                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0

            self.controller_step += 1
            if self.args.network_type == 'rnn' or self.args.network_type == 'cnn':
                valid_idx = (valid_idx + self.max_length) % (self.valid_data.size(0) - 1)

    def test_model(self, dag, name):
        #if self.args.network_type == 'classification':
        self.shared.eval()
        self.controller.eval()
        counter = 0
        if name == 'val':
            batcher = self.valid_data.get_batcher()
            num_steps = self.valid_data.num_steps
        elif name == 'test':
            batcher = self.test_data.get_batcher()
            num_steps = self.test_data.num_steps
        else:
            raise Exception("Unknown {}".format(name))

        if self.args.network_type == 'classification' and self.args.dataset in ['multitask', 'fnc', 'qnli', 'rte', 'wnli', 'multitask_glue']:
            true_label = []
            pred_label = []
            for i in range(num_steps):
                batch = next(batcher)
                if self.args.use_elmo and not self.args.use_precomputed_elmo:
                    seq1 = batch.get('original_seq1')
                    seq1_list = []
                    for s in seq1:
                        seq1_list.append(s.split(' '))
                    seq1 = self.elmo(batch_to_ids(seq1_list).cuda())
                    seq1_len = seq1['mask'].sum(1)
                    seq1 = seq1['elmo_representations'][0]

                    seq2 = batch.get('original_seq2')
                    seq2_list = []
                    for s in seq2:
                        seq2_list.append(s.split(' '))

                    seq2 = self.elmo(batch_to_ids(seq2_list).cuda())
                    seq2_len = seq2['mask'].sum(1)
                    seq2 = seq2['elmo_representations'][0]
                else:
                    seq1 = batch.get('seq1_batch')
                    seq1_len = batch.get('seq1_length')
                    seq1 = to_var(self.args, seq1)
                    seq2 = batch.get('seq2_batch')
                    seq2_len = batch.get('seq2_length')
                    seq2 = to_var(self.args, seq2)
                label = batch.get('label')
                label = to_var(self.args, label)
                logits, probs, preds = self.shared.forward_nas(seq1, seq1_len, seq2, seq2_len, dag)
                true_label.extend(label.cpu().data.numpy().tolist())
                pred_label.extend(preds.cpu().data.numpy().tolist())


            # order of labels is AGAINST, NONE, FAVOUR
            if self.args.dataset in ['stance', 'target', 'multitask']:
                f1scores = f1_score(true_label, pred_label, average=None)
                f1score = np.mean([f1scores[0], f1scores[2]])
                self.tb.scalar_summary(f"test/{name}_F1", f1score, self.epoch)
                logger.info(f"{name} F1 score: {f1score}")
            elif self.args.dataset in ['absa_laptop', 'absa_restaurant', 'fnc', 'qnli', 'rte', 'wnli', 'multitask_glue']:
                acc = 100.0 * (np.array(true_label)==np.array(pred_label)).sum()
                acc = acc/len(true_label)
                self.tb.scalar_summary(f"test/{name}_accuracy", acc, self.epoch)
                logger.info(f"{name} accuracy: {acc}")

            save_criteria_score = None
            if self.args.save_criteria == 'F1':
                save_criteria_score = f1score
            elif self.args.save_criteria == 'acc':
                save_criteria_score = acc
            return save_criteria_score
        


    def test_final_model(self):
        self.shared.eval()
        self.controller.eval()
        counter = 0
        if self.args.mode == 'retest':
            if self.args.load_dag:
                dag = self.get_best_dag()
            else:
                dag = t.load(os.path.join(self.args.model_dir,'best_dag.dat'))
        else:
            dag = self.derive()

        if self.test_data is None: # for GLUE tasks, there is no test set
            batcher = self.valid_data.get_batcher()
            num_steps = self.valid_data.num_steps
        else:
            batcher = self.test_data.get_batcher()
            num_steps = self.test_data.num_steps
        
        if self.args.network_type == 'classification' and self.args.dataset in ['multitask', 'qnli', 'rte', 'wnli', 'multitask_glue']:
            true_label = []
            pred_label = []
            for i in range(num_steps):
                batch = next(batcher)
                #ipdb.set_trace()
                seq1 = batch.get('seq1_batch')
                seq1_len = batch.get('seq1_length')
                seq1 = to_var(self.args, seq1)
                seq2 = batch.get('seq2_batch')
                seq2_len = batch.get('seq2_length')
                seq2 = to_var(self.args, seq2)
                label = batch.get('label')
                label = to_var(self.args, label)
                logits, probs, preds = self.shared.forward_nas(seq1, seq1_len, seq2, seq2_len, dag)
                true_label.extend(label.cpu().data.numpy().tolist())
                pred_label.extend(preds.cpu().data.numpy().tolist())

            # order of labels is AGAINST, NONE, FAVOUR
            if self.args.dataset in ['stance', 'target', 'multitask']:
                f1scores = f1_score(true_label, pred_label, average=None)
                f1score = np.mean([f1scores[0], f1scores[2]])
                logger.info(f"test F1 score: {f1score}")

            
            elif self.args.dataset in ['qnli', 'rte', 'wnli', 'multitask_glue']:

                if self.args.mode == 'retest' and self.test_data is not None: # glue hidden test set evaluation
                    if not os.path.exists(os.path.join(self.args.model_dir,'results')):
                        os.mkdir(os.path.join(self.args.model_dir,'results'))
                    
                    file_loc = self.create_result_shared_path(filename=f'test_{self.args.dataset}.tsv')
                    logger.info(f'Writing the test results to {file_loc}')
                    fw = open(file_loc,'w')
                    fw.write('index\tprediction\n')
                    for ind, l in enumerate(pred_label):
                        if l == 0:
                            if self.args.dataset in ['wnli']:
                                fw.write(str(ind)+'\t'+'0\n')
                            elif self.args.dataset in ['qnli','rte']:
                                fw.write(str(ind)+'\t'+'entailment\n')
                        else:
                            if self.args.dataset in ['wnli']:
                                fw.write(str(ind)+'\t'+'1\n')
                            elif self.args.dataset in ['qnli', 'rte']:
                                fw.write(str(ind)+'\t'+'not_entailment\n')

                        fw.flush()

                    fw.close()
                else:
                    acc = 100.0 * (np.array(true_label)==np.array(pred_label)).sum()
                    acc = acc/len(true_label)
                    logger.info(f"test accuracy: {acc}")

            if self.test_data is None:
                # save the result
                logger.info("saving the result...")
                if not os.path.exists(os.path.join(self.args.model_dir,'results')):
                    os.mkdir(os.path.join(self.args.model_dir,'results'))
                result_save_path = self.result_shared_path
                final_dict = {}
                final_dict['args'] = self.args.__dict__
                if self.args.dataset in ['absa_laptop', 'absa_restaurant', 'fnc', 'qnli', 'rte', 'wnli', 'multitask_glue']:
                    final_dict['scores'] = acc
                elif self.args.dataset in ['stance', 'target', 'multitask']:
                    final_dict['scores'] = f1score
                with open(result_save_path, 'w') as fp:
                    json.dump(final_dict, fp, indent=4, sort_keys=True)


                file_loc = self.create_result_shared_path(filename='true_labels.txt')
                print(file_loc)
                with open(file_loc, 'w') as f:
                    f.write('\n'.join([str(l) for l in true_label]))
                file_loc = self.create_result_shared_path(filename='pred_labels.txt')
                with open(file_loc, 'w') as f:
                    f.write('\n'.join([str(l) for l in pred_label]))

                file_loc = self.create_result_shared_path(filename='correct_scores.txt')
                with open(file_loc, 'w') as f:
                    f.write('\n'.join([str(1.0) if l else str(0.0) for l in (np.array(true_label)==np.array(pred_label)).tolist()]))



    def get_l2_regularization(self):
        l_reg = None
        for W in self.shared.parameters():
            if l_reg is None:
                l_reg  = W.norm(p=2)
            else:
                l_reg = l_reg + W.norm(p=2)


        return self.args.l2_reg_lambda*l_reg


    def get_block_sparse_regularization(self):
        l_reg = None
        for W in self.shared.parameters():
            if l_reg is None:
                if W.dim() == 1:
                    l_reg = W.norm(p=1)
                else:
                    l_reg = W.norm(p=2, dim=1).sum()
            else:
                if W.dim() == 1:
                    l_reg = l_reg + W.norm(p=1)
                else:
                    l_reg = l_reg + W.norm(p=2, dim=1).sum()

        return self.args.block_sparse_reg_lambda*l_reg


    def get_condition2_regularization(self):        
        # read the dag nodes 
        dag = self.dag_ref
        node_map = self.get_node_map(self.dag_ref)

        l_reg = None
        d_reg = None
        params1 = []

        for name, param in self.shared.named_parameters():
            #print(name)
            if 'w_xh' in name or 'w_ch' in name or 'w_m1' in name or 'w_m0' in name:
                params1.append(param)
            if '_w_h' in name or '_w_c' in name:
                for node in node_map:
                    if '_w_h.'+str(node) in name or '_w_c.'+str(node) in name:
                        params1.append(param)
            if not ('w_xh' in name or 'w_ch' in name or 'w_m1' in name or 'w_m0' in name or '_w_h' in name or '_w_c' in name):
                params1.append(param)

        params2 = []

        for name, param in self.shared_ref.named_parameters():
            #print(name)
            if 'w_xh' in name or 'w_ch' in name or 'w_m1' in name or 'w_m0' in name:
                params2.append(param)
            if '_w_h' in name or '_w_c' in name:
                for node in node_map:
                    if '_w_h.'+str(node) in name or '_w_c.'+str(node) in name:
                        params2.append(param)

            if not ('w_xh' in name or 'w_ch' in name or 'w_m1' in name or 'w_m0' in name or '_w_h' in name or '_w_c' in name):
                params2.append(param)


        for p1,p2 in zip(params1, params2):
            if l_reg is None:
                if (p1-p2).dim() == 1:
                    l_reg = (p1-p2).norm(p=1)
                else:
                    l_reg = (p1-p2).norm(p=2, dim=1).sum()
            else:
                if (p1-p2).dim() == 1:
                    l_reg = l_reg + (p1-p2).norm(p=1)
                else:
                    l_reg = l_reg + (p1-p2).norm(p=2, dim=1).sum()

            if p1.dim() == 1:
                ortho = (p1-p2)*p2
            else:
                ortho = t.mm(t.t(p1-p2),p2)
            if d_reg is None:
                d_reg = ortho.norm(p=2)
            else:
                d_reg = d_reg + ortho.norm(p=2)

        return self.args.block_sparse_reg_lambda*l_reg+self.args.orthogonal_reg_lambda*d_reg

 
    def get_edges_mapper(self, num_blocks):
        mapper = {}
        counter = 0
        if self.args.nas_type == 'NAS_RNN':
            for idx in range(num_blocks):
                for jdx in range(idx+1,num_blocks):
                    mapper[str(idx)+str(jdx)] = counter
                    counter += 1
        return mapper
        

    def get_node_map(self, dag):
        
        edges_mapper = self.get_edges_mapper(self.args.num_blocks)
        edges = []
        if self.args.nas_type == 'NAS_RNN':
            leaf_node_ids = []
            q = deque()
            q.append(0)

            while True:
                if len(q) == 0:
                    break
                node_id = q.popleft()
                nodes = dag[node_id]
                for next_node in nodes:
                    next_id = next_node.id
                    if next_id == self.args.num_blocks:
                        leaf_node_ids.append(node_id)
                        assert len(nodes) == 1, "parent of leaf node should have only one child"
                        continue
                    edges.append(edges_mapper[str(node_id)+str(next_id)])
                    q.append(next_id)

        return edges


    def derive(self, sample_num=None, valid_idx=0):
        if sample_num is None:
            sample_num = self.args.derive_num_sample
        

        dags, log_probs, entropies = self.controller.sample(sample_num, with_details=True)


        max_R, best_dag = -1.0, None
        pbar = tqdm(dags, desc="derive")
        for dag in pbar:
            #logger.info(f'{dag}')
            if self.args.network_type in ['seq2seq','classification']:
                R = self.get_reward([dag], entropies)
            else:
                raise Exception(f"Unknown network type : self.args.network_type")

            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag
            pbar.set_description(f"derive| max_R: {max_R:8.6f}")
        

        fname = f"{self.epoch:03d}-{self.controller_step:06d}-{max_R:6.4f}-best.png"
        path = os.path.join(self.args.model_dir, "networks", fname)
        
        
        if self.current_best_dag(max_R):
            draw_network(best_dag, path)
            self.tb.image_summary("derive/best", [path], self.epoch)
        else:
            draw_network(best_dag, path)

        dag_name =  f"{self.epoch:03d}-{self.controller_step:06d}-{max_R:6.4f}-best-dag.dat"

        if self.args.mode == 'train':
            logger.info(f"Saving the best dag as {dag_name}")
            t.save(best_dag, os.path.join(self.args.model_dir,"networks",dag_name))

        return best_dag

    def current_best_dag(self, R):
        paths = glob(os.path.join(self.args.model_dir, 'networks', '*-best.png'))
        if len(paths) == 0:
            return True
        else:
            for path in paths:
                score = float(path.split('-')[-2])
                if R < score:
                    return False

            return True



    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.shared.parameters() if p.requires_grad)


    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.decay_after + 1, 0)
        return self.args.lr * (self.args.decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr

    def create_result_shared_path(self, filename):
        return f'{self.args.model_dir}/results/shared_epoch{self.epoch}_step{self.shared_step}_{filename}'

    def get_batch(self, source, idx, length=None, evaluation=False):
        # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        length = min(length if length else self.max_length, len(source) - 1 - idx)
        data = Variable(source[idx:idx+length], volatile=evaluation)
        target = Variable(source[idx+1:idx+1+length].view(-1))
        return data, target

    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared_epoch{self.epoch}_step{self.shared_step}.pth'

    @property
    def controller_path(self):
            return f'{self.args.model_dir}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def result_shared_path(self):
        return f'{self.args.model_dir}/results/shared_epoch{self.epoch}_step{self.shared_step}.json'

    @property
    def output_result_shared_path(self):
        return f'{self.args.model_dir}/results/output_shared_epoch{self.epoch}_step{self.shared_step}.txt'


    def get_saved_models_info(self, load_path=''):
        if load_path:
            paths = glob(os.path.join(load_path, '*.pth'))
        else:
            paths = glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def get_best_dag(self, load_path=None):

        if self.args.load_dag is not None and '.dat' in self.args.load_dag:
            load_path = self.args.load_dag

        if load_path is None:
            if self.args.load_dag:
                paths = glob(os.path.join(self.args.load_dag,'networks','*.dat'))
            else:
                paths = glob(os.path.join(self.args.load_path,'networks' ,'*.dat'))

        
            best_dag_path = ''
            best_score = 0.0
            for path in paths:
                basename = os.path.basename(path)
                #print(basename)
                score = float(basename.split('-')[2])
                if score >= best_score:
                    best_score = score
                    best_dag_path = path
        else:
            best_dag_path = load_path

        print(best_dag_path)
        best_dag = t.load(best_dag_path)
        logger.info(f"[*] LOADED: {best_dag_path}")

        if self.args.nas_type == 'NAS_RNN':
            self.print_dag(best_dag)

        return best_dag


    def print_dag(self, dag):
        leaf_node_ids = []
        q = deque()
        q.append(0)
        print('DAG STRUCTURE:')
        print(f"Node: 0, Input Node: x[t], h[t], Function: {dag[-1][0].name}")
        while True:
            if len(q) == 0:
                break
            node_id = q.popleft()
            nodes = dag[node_id]
            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, "parent of leaf node should have only one child"
                    continue
                print(f"Node: {next_id}, Input Node: {node_id}, Function:{next_node.name}")

                q.append(next_id)



    def save_model(self, save_criteria_score=None):
        t.save(self.shared.state_dict(), self.shared_path)
        logger.info(f"[*] SAVED: {self.shared_path}")
        
        t.save(self.controller.state_dict(), self.controller_path)
        logger.info(f"[*] SAVED: {self.controller_path}")

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if save_criteria_score is not None:
            if os.path.exists(os.path.join(self.args.model_dir,'checkpoint_tracker.dat')):
                checkpoint_tracker = t.load(os.path.join(self.args.model_dir,'checkpoint_tracker.dat'))

            else:
                checkpoint_tracker = {}
            key = f"{self.epoch}_{self.shared_step}"
            value = save_criteria_score
            checkpoint_tracker[key] = value
            if len(epochs)>=self.args.max_save_num:
                low_value = 100000.0
                remove_key = None
                for key,value in checkpoint_tracker.items():
                    if low_value > value:
                        remove_key = key
                        low_value = value

                del checkpoint_tracker[remove_key]

                remove_epoch = remove_key.split("_")[0]
                paths = glob(os.path.join(self.args.model_dir,f'*_epoch{remove_epoch}_*.pth'))
                for path in paths:
                    remove_file(path)

            # save back the checkpointer tracker
            t.save(checkpoint_tracker, os.path.join(self.args.model_dir,'checkpoint_tracker.dat'))

                    

        else:
            for epoch in epochs[:-self.args.max_save_num]:
                paths = glob(os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))

                for path in paths:
                    remove_file(path)

    def remove_gradient_update(self, model):
        for W in model.parameters():
            W.requires_grad = False

    def load_model(self, load_path):
        epochs, shared_steps, controller_steps = self.get_saved_models_info(load_path)

        if len(epochs) == 0:
            logger.info(f"[!] No checkpoint found in {self.args.model_dir}...")
            return

        if os.path.exists(os.path.join(load_path, 'checkpoint_tracker.dat')):
            checkpoint_tracker = t.load(os.path.join(load_path,'checkpoint_tracker.dat'))
            best_key = None
            best_score = -1.0
            for key,value in checkpoint_tracker.items():
                if value>best_score:
                    best_score = value
                    best_key = key

            self.epoch = int(best_key.split("_")[0])
            self.shared_step = int(best_key.split("_")[1])
            self.controller_step = max(controller_steps)
        else:     

            self.epoch = self.start_epoch = max(epochs)
            self.shared_step = max(shared_steps)
            self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.shared.load_state_dict(
                t.load(f'{load_path}/shared_epoch{self.epoch}_step{self.shared_step}.pth'
, map_location=map_location))
        logger.info(f"[*] LOADED: {load_path}/shared_epoch{self.epoch}_step{self.shared_step}.pth")
        if isinstance(self.controller,list):
            self.controller[0].load_state_dict(
                t.load(f'{load_path}/controller_epoch{self.epoch}_step{self.controller_step}.pth', map_location=map_location))
        else:
            self.controller.load_state_dict(
                t.load(f'{load_path}/controller_epoch{self.epoch}_step{self.controller_step}.pth', map_location=map_location))
        logger.info(f"[*] LOADED: {load_path}/controller_epoch{self.epoch}_step{self.controller_step}.pth")

