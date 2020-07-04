import pandas as pd
from mosestokenizer import MosesDetokenizer
from scipy.stats import pearsonr   
def pearson(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    return '{0:.{1}f}'.format(pearson_corr, 3)

reference_list = dict({
        "cs-en": 'testset_cs-en.tsv',
        "de-en": 'testset_de-en.tsv',
        "fi-en": 'testset_fi-en.tsv',
        "lv-en": 'testset_lv-en.tsv',
        "ru-en": 'testset_ru-en.tsv',
        "tr-en": 'testset_tr-en.tsv',
        "zh-en": 'testset_zh-en.tsv',
        })

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
parser.add_argument('--do_lower_case', type=bool, default=False)
parser.add_argument('--language_model', type=str, default='gpt2')
parser.add_argument('--mapping', type=str, default='CLP', help='CLP or UMD')    

import json
args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))

from scorer import XMOVERScorer
import numpy as np
import torch
import truecase

scorer = XMOVERScorer(args.model_name, args.language_model, args.do_lower_case)

def metric_combination(a, b, alpha):
    return alpha[0]*np.array(a) + alpha[1]*np.array(b)

import os
from tqdm import tqdm
for pair in tqdm(reference_list.items()):
    lp, path = pair
    src, tgt = lp.split('-')
    
    device='cuda:0'
    
    temp = np.loadtxt('mapping/europarl-v7.' + src + '-' + tgt + '.2k.12.BAM.map')
    projection = torch.tensor(temp, dtype=torch.float).to(device)
    
    temp = np.loadtxt('mapping/europarl-v7.' + src + '-' + tgt + '.2k.12.GBDD.map')
    bias = torch.tensor(temp, dtype=torch.float).to(device)
    
    data = pd.read_csv(os.path.join('testset', path), sep='\t') 
    references = data['reference'].tolist()
    translations = data['translation'].tolist()
    source = data['source'].tolist()
    human_score = data['HUMAN_score'].tolist()
    sentBLEU = data['sentBLEU'].tolist()

    with MosesDetokenizer(src) as detokenize:        
        source = [detokenize(s.split(' ')) for s in source]         
    with MosesDetokenizer(tgt) as detokenize:                
        references = [detokenize(s.split(' ')) for s in references]        
        translations = [detokenize(s.split(' ')) for s in translations]

    translations = [truecase.get_true_case(s) for s in translations]
    
    xmoverscores = scorer.compute_xmoverscore(args.mapping, projection, bias, source, translations, ngram=2, bs=64)
    
    lm_scores = scorer.compute_perplexity(translations, bs=1)
    
    scores = metric_combination(xmoverscores, lm_scores, [1, 0.1])
    
    print('\r\nlp:{} xmovescore:{} xmoverscore+lm:{}'.format(lp, 
                                      pearson(human_score, xmoverscores),
                                      pearson(human_score, scores)))
