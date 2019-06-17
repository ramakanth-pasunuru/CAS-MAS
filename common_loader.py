from __future__ import print_function
import json
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import ipdb
import h5py
np.random.seed(111)




PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[END]' # This has a vocab id, which is used at the end of untruncated target sequences


def load_glove_emb(glove_file_path, vocab):

    basename = os.path.basename(glove_file_path)
    dim = int(basename.split('.')[2][:-1])
    random_inits = np.random.normal(0.0,1.0,[100,dim]) # keep 100 random vectors for unknown words
    wemb = np.random.normal(0.0,1.0,[vocab.size(),dim])

    glove_dict = {}

    counter = 0

    with open(glove_file_path,'r') as f:
        for line in f.read().splitlines():
            line = line.split(' ')
            word = line[0]
            feats = line[1:]
            feats = [float(feat) for feat in feats]
            glove_dict[word] = feats

    for i in range(vocab.size()):
        word = vocab.id2word(i)
        feat = glove_dict.get(word)
        if feat is not None:
            wemb[i,:] = feat
            counter += 1
        else:
            index = counter%100
            wemb[i,:] = random_inits[index,:]
    
    return torch.FloatTensor(wemb)




class Vocab(object):
    # https://github.com/abisee/pointer-generator/blob/master/data.py
    def __init__(self,vocab_file,max_size):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
            Args:
                vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. 
                            This code doesn't actually use the frequencies, though.
                max_size: integer. The maximum size of the resulting Vocabulary.
                
        """
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # [PAD], [START], [STOP] and [UNK] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, START_DECODING, STOP_DECODING, UNKNOWN_TOKEN]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f.read().splitlines():
                pieces = line.split('\t')
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def print_id2word(self):
        print(self._id_to_word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def texttoidx(self,text,max_sentence_length, add_start_token=False):
        text = text + ' ' + STOP_DECODING
        if add_start_token:
            text = START_DECODING + ' ' + text
        tokens = []
        seq_length = 0
        for word in text.split()[:max_sentence_length]: # also need one more for [END] token
            tokens.append(self.word2id(word))
            seq_length += 1

        tokens.extend([0 for i in range(max_sentence_length-len(tokens))])

        return np.asarray(tokens),seq_length


class Batch(object):

    def __init__(self):
        self._dict = {}


    def put(self,key,value):
        if self._dict.get(key) is None:
            self._dict[key] = value
        else:
            raise Exception("key:{} already exits".format(key))

    def get(self,key):
       if self._dict.get(key) is not None:
           return self._dict[key]
       else:
           raise Exception("key:{} does not exits".format(key))



class TextClassificationBatcher(object):

    def __init__(self,hps,mode,vocab, task_name=None):
        self._data_path = hps.data_dir
        self._max_steps = hps.rnn_max_length
        self.transfer_learning = hps.transfer_learning
        self._mode = mode
        self._batch_size = hps.batch_size
        self.vocab = vocab
        self._use_precomputed_elmo = hps.use_precomputed_elmo
        self.task_name = task_name
        if self.task_name is not None:
            self._data_path = os.path.join(self._data_path, task_name)
        if self.transfer_learning and self._mode != 'train':
            self._data_path = hps.tl_data_path
        self.data  = self._process_data()
        self.num_steps = int(len(self.data)/self._batch_size) + 1

    def _process_data(self):
        data = []
       
        filename1 = self._mode+'.sequence_1'
        filename2 = self._mode+'.sequence_2'
        
        filename_label = self._mode+'.labels'

        seq1 = []
        seq2 = []
        label = []
        seq1_elmo = []
        seq2_elmo = []
        with open(os.path.join(self._data_path,filename1),'r') as f:
            for ind, line in enumerate(f.read().splitlines()):
                seq1.append(line.strip())
                seq1_elmo.append(str(ind))

        with open(os.path.join(self._data_path,filename2),'r') as f:
            for ind,line in enumerate(f.read().splitlines()):
                seq2.append(line.strip())
                seq2_elmo.append(str(ind))

        with open(os.path.join(self._data_path,filename_label),'r') as f:
            for line in f.read().splitlines():
                label.append(line.strip())

        if self._use_precomputed_elmo:
            data = list(zip(seq1, seq2, seq1_elmo, seq2_elmo, label))
        else:
            data = list(zip(seq1,seq2,label))
      
        if self._mode == 'train':
            np.random.shuffle(data)

        return data 

    
    def get_batcher(self):
        """
        This module process data and creates batches for train/val/test 
        Also acts as generator
        """
        if self._mode == 'train':
            np.random.shuffle(self.data)

        label_ref = {}

        if self._use_precomputed_elmo:
            hdf5_seq1_data = h5py.File(os.path.join(self._data_path,'elmo_'+self._mode+'.sequence_1.hdf5'), 'r')
            hdf5_seq2_data = h5py.File(os.path.join(self._data_path, 'elmo_'+self._mode+'.sequence_2.hdf5'), 'r')


        with open(os.path.join(self._data_path,'train.label_vocab'), 'r') as f:
            for ind,line in enumerate(f.read().splitlines()):
                label_ref[line] = ind
                
        
        for i in range(0,len(self.data),self._batch_size):
            start = i
            if i+self._batch_size > len(self.data): # handling leftovers
                end = len(self.data)
                current_batch_size = end-start
            else:
                end = i+self._batch_size
                current_batch_size = self._batch_size

            if self._use_precomputed_elmo:
                original_seq1, original_seq2, seq1_ids, seq2_ids, label = zip(*self.data[start:end])
            else:
                original_seq1, original_seq2, label = zip(*self.data[start:end])

            seq1_batch = []
            seq1_length = []
            seq2_batch = []
            seq2_length = []

            if self._use_precomputed_elmo:
                for s1, s2 in zip(seq1_ids, seq2_ids):
                    d1 = hdf5_seq1_data.get(s1)
                    d2 = hdf5_seq2_data.get(s2)
                    d1_len = d1.shape[1]
                    d2_len = d2.shape[1]
                    if d1_len<=self._max_steps:
                        seq1_batch.append(np.concatenate((d1,np.zeros((3,self._max_steps-d1_len,1024))),axis=1))
                        seq1_length.append(d1_len)
                    else:
                        seq1_batch.append(d1[:,:self._max_steps,:])
                        seq1_length.append(self._max_steps)

                    if d2_len<=self._max_steps:
                        seq2_batch.append(np.concatenate((d2,np.zeros((3,self._max_steps-d2_len,1024))),axis=1))
                        seq2_length.append(d2_len)
                    else:
                        seq2_batch.append(d2[:,:self._max_steps,:])
                        seq2_length.append(self._max_steps)

            else:
            
                for s1, s2 in zip(original_seq1, original_seq2):
                    s1_id, s1_length = self.vocab.texttoidx(s1, self._max_steps, add_start_token=True)
                    s2_id, s2_length = self.vocab.texttoidx(s2, self._max_steps, add_start_token=True)
                    seq1_batch.append(s1_id)
                    seq1_length.append(s1_length)
                    seq2_batch.append(s2_id)
                    seq2_length.append(s2_length)

            label = list(map(lambda l: label_ref[l] if label_ref.get(l) is not None else 0,label)) #add special case to handle test

            batch = Batch()
            batch.put('original_seq1', original_seq1)
            batch.put('original_seq2', original_seq2)
            if self._use_precomputed_elmo:
                batch.put('seq1_batch', torch.FloatTensor(np.asarray(seq1_batch)))
                batch.put('seq2_batch', torch.FloatTensor(np.asarray(seq2_batch)))
            else:
                batch.put('seq1_batch', torch.LongTensor(np.asarray(seq1_batch)))
                batch.put('seq2_batch', torch.LongTensor(np.asarray(seq2_batch)))
            batch.put('seq1_length', np.asarray(seq1_length)) 
            batch.put('seq2_length', np.asarray(seq2_length))
            batch.put('label', torch.LongTensor(np.asarray(label)))
            yield batch


      





            






