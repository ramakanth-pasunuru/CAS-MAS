#!/usr/bin/env python
import torch
import copy
from common_loader import *
import models
import config
from utils import *
from trainer import TrainerNAS, Trainer
from utils import get_logger

logger = get_logger()


def main(args):
    prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'classification':
        vocab = Vocab(args.vocab_file, args.max_vocab_size)
        dataset = {}

        if args.dataset in ['qnli', 'rte', 'wnli']:
            dataset['train'] = TextClassificationBatcher(args, 'train', vocab)
            dataset['val'] = TextClassificationBatcher(args, 'val', vocab)
            if args.dataset not in ['qnli', 'rte', 'wnli']:
            #if args.dataset not in []: # uncomment this and comment above for glue test outputs
                dataset['test'] = TextClassificationBatcher(args, 'test', vocab)
            else:
                dataset['test'] = None
        elif args.dataset in ['multitask', 'multitask_glue']:
            dataset['train'] = {}
            dataset['test'] = {}
            dataset['val'] = {}
            for task in args.multitask:
                dataset['train'][task] = TextClassificationBatcher(args, 'train', vocab, task_name=task)
                dataset['val'][task] = TextClassificationBatcher(args, 'val', vocab, task_name=task)
                
                if task not in ['qnli', 'rte', 'wnli']:
                    dataset['test'][task] = TextClassificationBatcher(args, 'test', vocab, task_name=task)
        else:
            raise Exception(f"Unknown dataset: {args.dataset} for the corresponding network type: {args.network_type}")


    else:
        raise NotImplemented(f"{args.dataset} is not supported")

    if args.nas:
        trainer = TrainerNAS(args, dataset)
    else:
        trainer = Trainer(args, dataset)

    if args.mode == 'train':
        save_args(args)
        trainer.train()
    elif args.mode == 'retrain':
        trainer.retrain()
    elif args.mode == 'derive':
        assert args.load_path != "", "`--load_path` should be given in `derive` mode"
        trainer.derive()
    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        if args.nas:
            trainer.test_final_model()
        else:
            trainer.test(args.mode)

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)

