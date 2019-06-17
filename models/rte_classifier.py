import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import numpy as np
import bottleneck as bn
from models.nas_rnn import NAS_RNN



nas_cell = {'NAS_RNN':NAS_RNN}

def rnn_mask(seq_lens, max_step, cuda):
    """
    Creates a mask for variable length sequences
    """
    num_batches = len(seq_lens)

    mask = torch.FloatTensor(num_batches, max_step).zero_()
    if cuda:
        mask = mask.cuda()
    for b, batch_l in enumerate(seq_lens):
        mask[b, :batch_l] = 1.0
    mask = Variable(mask)
    return mask

class RTEClassifier(nn.Module):
    def __init__(self, args):
        super(RTEClassifier, self).__init__()
        self.enable_cuda = args.cuda
        self.embed_size = args.embed
        self.hidden_dim = args.hid
        self.vocab_size = args.max_vocab_size
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.nas = args.nas
        self.multitask = args.multitask
        self.args = args
        self.num_blocks = args.num_blocks
        self.birnn = args.birnn

        if self.args.use_precomputed_elmo:
            self.scalar_parameters = nn.ParameterList(
                                    [nn.Parameter(torch.FloatTensor([1.0]),
                                    requires_grad=True) for i in range(3)])
            self.gamma = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        if self.args.use_elmo:
            self.embed = nn.Linear(1024, self.embed_size)
        elif self.args.use_bert:
            self.embed = nn.Linear(4*768, self.embed_size)
        else:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        if not self.nas:
            self.encoder = nn.LSTM(self.embed_size, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.birnn, dropout=args.dropout)
        else:
            if self.birnn:
                self.encoder_fw = nas_cell[self.args.nas_type](args, self.embed_size, self.hidden_dim, self.num_blocks)
                self.encoder_bw = nas_cell[self.args.nas_type](args, self.embed_size, self.hidden_dim, self.num_blocks)
            else:
                self.encoder = nas_cell[self.args.nas_type](args, self.embed_size, self.hidden_dim, self.num_blocks)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.seq_count = 2 if self.birnn else 1
        self.mlp_layer = nn.Linear(self.seq_count*4*self.hidden_dim, self.hidden_dim)
        if self.multitask is None:
            if self.args.use_cas:
                self.proj_layer = {}
                self.proj_layer['qnli'] = nn.Linear(self.hidden_dim, self.num_classes)
                self.proj_layer['wnli'] = nn.Linear(self.hidden_dim, self.num_classes)
                self.proj_layer['rte'] = nn.Linear(self.hidden_dim, self.num_classes)
                self._proj_layer = nn.ModuleList([v for k,v in self.proj_layer.items()])

            else:
                self.proj_layer = nn.Linear(self.hidden_dim, self.num_classes)
        else:
            self.proj_layer = {}
            for task in self.multitask:    
                self.proj_layer[task] = nn.Linear(self.hidden_dim, self.num_classes)

            self._proj_layer = nn.ModuleList([v for k,v in self.proj_layer.items()])

        if self.args.weight_init:
            self.init_weights()


    def init_weights(self):
        if not self.nas:
            self.encoder.weight_hh_l0.data.uniform_(-self.args.weight_init, self.args.weight_init)
            self.encoder.weight_ih_l0.data.uniform_(-self.args.weight_init, self.args.weight_init)
            self.encoder.bias_ih_l0.data.fill_(0)
            self.encoder.bias_hh_l0.data.fill_(0)

        self.proj_layer.weight.data.uniform_(-self.args.weight_init, self.args.weight_init)
        self.mlp_layer.weight.data.uniform_(-self.args.weight_init, self.args.weight_init)





    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.seq_count, batch_size, self.hidden_dim)),
                    Variable(torch.zeros(self.seq_count, batch_size, self.hidden_dim)))
    

    def scalar_mix(self, sent_batch):
        normed_weights = F.softmax(torch.cat([parameter for parameter
                                                                in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        return self.gamma*(normed_weights[0]*sent_batch[:,0,:,:] + 
                            normed_weights[1]*sent_batch[:,1,:,:] + 
                            normed_weights[2]*sent_batch[:,2,:,:]) 

    def forward(self, premise, premise_len, hypothesis, hypothesis_len):
        if self.args.use_precomputed_elmo:
            premise = self.scalar_mix(premise)
            hypothesis = self.scalar_mix(hypothesis)

        if self.args.use_elmo or self.args.use_bert:
            batch_size, step_size,_ = premise.shape
        else:
            batch_size, step_size = premise.shape

        premise_emb = self.embed(premise)
        hypo_emb = self.embed(hypothesis)

        # add a dropout layer
        premise_emb = self.dropout(premise_emb)
        hypo_emb = self.dropout(hypo_emb)


        premise_outputs, (ht, ct) = self.encoder(premise_emb)
        mask_p = rnn_mask(premise_len,step_size, self.enable_cuda)
        premise_outputs = premise_outputs * mask_p.unsqueeze(2).expand_as(premise_outputs)

        hypo_outputs, (ht, ct) = self.encoder(hypo_emb)
        mask_h = rnn_mask(hypothesis_len, hypothesis.shape[1], self.enable_cuda)
        hypo_outputs = hypo_outputs * mask_h.unsqueeze(2).expand_as(hypo_outputs)
        # do max pooling
        if self.args.model_type == 'max-pool':
            premise_rep = torch.max(premise_outputs, 1)[0]
            hypo_rep = torch.max(hypo_outputs, 1)[0]
        elif self.args.model_type == 'selfatten':    
            premise_rep,_ = self.atten(premise_outputs, mask_p)
            hypo_rep,_ = self.atten(hypo_outputs, mask_h)


        final_rep = torch.cat((premise_rep, hypo_rep, premise_rep-hypo_rep, premise_rep * hypo_rep), 1)

        # add multilayer perceptron
        mlp_out = self.mlp_layer(final_rep)
        mlp_out = self.relu(mlp_out)
        mlp_out = self.dropout(mlp_out)

        logits = self.proj_layer(mlp_out)
        probs = F.softmax(logits, 1)
        pred = torch.max(probs, 1)[1]

        return logits, probs, pred

    def birnn_encoder(self, input_embs, dag):
        
        batch_size, step_size, _ = input_embs.shape
        if self.args.hidden_varient == 'lstm':
            ht_fw = [Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda(),
                    Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda()]
            ht_bw = [Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda(),
                    Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda()]
        else:
            ht_fw = Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda()
            ht_bw = Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda()

        
        outputs_fw = []
        outputs_bw = []

        for i in range(step_size):
            output, ht_fw = self.encoder_fw(input_embs[:,i,:], ht_fw, dag)
            outputs_fw.append(output)
            output, ht_bw = self.encoder_bw(input_embs[:,step_size-i-1], ht_bw, dag)
            outputs_bw.append(output)

        # reverse the outputs_bw to properly concat with the outputs_fw
        outputs_bw = outputs_bw[::-1]

        outputs_fw = torch.transpose(torch.stack(outputs_fw), 0, 1)
        outputs_bw = torch.transpose(torch.stack(outputs_bw), 0, 1)
        

        outputs = torch.cat((outputs_fw, outputs_fw), 2)

        return outputs


    def forward_nas(self, premise, premise_len, hypothesis, hypothesis_len,  dag):

        if self.args.use_precomputed_elmo:
            premise = self.scalar_mix(premise)
            hypothesis = self.scalar_mix(hypothesis)


        if self.args.use_elmo or self.args.use_bert:
            batch_size, step_size,_ = premise.shape
        else:
            batch_size, step_size = premise.shape

        premise_emb = self.embed(premise)
        hypo_emb = self.embed(hypothesis)

        # add a dropout layer
        premise_emb = self.dropout(premise_emb)
        hypo_emb = self.dropout(hypo_emb)

        ht = Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda()
    
        if self.birnn:
            if isinstance(dag, list):
                premise_outputs = self.birnn_encoder(premise_emb, dag[0])
            else:
                premise_outputs = self.birnn_encoder(premise_emb, dag)

        else:
            premise_outputs = []
            for i in range(step_size):
                if isinstance(dag, list):
                    output, ht = self.encoder(premise_emb[:,i,:], ht, dag[0])
                else:
                    output, ht = self.encoder(premise_emb[:,i,:], ht, dag)

                premise_outputs.append(output)
            premise_outputs = torch.transpose(torch.stack(premise_outputs), 0, 1) 
            # converting from step_size x batch_size x hidden_size to batch_size x step_size x hidden_size

        mask_p = rnn_mask(premise_len,step_size, self.enable_cuda)
        premise_outputs = premise_outputs * mask_p.unsqueeze(2).expand_as(premise_outputs)



        if self.birnn:
            if isinstance(dag, list):
                if self.args.use_dual_controller:
                    hypo_outputs = self.birnn_encoder(hypo_emb, dag[1])
                else:
                    hypo_outputs = self.birnn_encoder(hypo_emb, dag[0])
            else:
                hypo_outputs = self.birnn_encoder(hypo_emb, dag)
        else:
            hypo_outputs = []
            ht = Variable(torch.FloatTensor(batch_size,self.hidden_dim).zero_(), requires_grad=False).cuda()
            for i in range(step_size):
                if isinstance(dag, list):
                    if self.args.use_dual_controller:
                        output, ht = self.encoder(hypo_emb[:,i,:], ht, dag[1])
                    else:
                        output, ht = self.encoder(hypo_emb[:,i,:], ht, dag[0])
                else:
                    output, ht = self.encoder(hypo_emb[:,i,:], ht, dag)
                hypo_outputs.append(output)
            hypo_outputs = torch.transpose(torch.stack(hypo_outputs), 0, 1) 

        mask_h = rnn_mask(hypothesis_len, hypothesis.shape[1], self.enable_cuda)
        hypo_outputs = hypo_outputs * mask_h.unsqueeze(2).expand_as(hypo_outputs)

        
        # do max pooling
        if self.args.model_type == 'max-pool':
            premise_rep = torch.max(premise_outputs,1)[0]
            hypo_rep = torch.max(hypo_outputs,1)[0]
        elif self.args.model_type == 'selfatten':
            if isinstance(dag, list) and self.args.use_atten_controller:
                premise_rep,_ = self.atten(premise_outputs,mask_p,dag[-1])
                hypo_rep,_ = self.atten(hypo_outputs, mask_h, dag[-1])
            else:    
                premise_rep,_ = self.atten(premise_outputs, mask_p)
                hypo_rep,_ = self.atten(hypo_outputs, mask_h)

        final_rep = torch.cat((premise_rep, hypo_rep, premise_rep-hypo_rep, premise_rep * hypo_rep), 1)

        # add multilayer perceptron

        mlp_out = self.mlp_layer(final_rep)
        mlp_out = self.relu(mlp_out)
        mlp_out = self.dropout(mlp_out)
        if self.multitask is None:
            if self.args.use_cas:
                logits = self.proj_layer[self.args.dataset](mlp_out)
            else:
                logits = self.proj_layer(mlp_out)
            probs = F.softmax(logits, 1)
            pred = torch.max(probs, 1)[1]
        else:
            logits = {}
            probs = {}
            pred = {}
            for key, value in self.proj_layer.items():
                logits[key] = value(mlp_out)
                probs[key] = F.softmax(logits[key],1)
                pred[key] = torch.max(probs[key], 1)[1]

        return logits, probs, pred


