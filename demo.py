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
parser.add_argument('--alignment', type=str, default='CLP', help='CLP or UMD or None')    
parser.add_argument('--ngram', type=int, default=2)
parser.add_argument('--layer', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Remove the percentage of noisy elements in Word-Mover-Distance')

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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    temp = np.load('mapping/layer-8/europarl-v7.%s-%s.%s.BAM' % (src, tgt, args.layer), allow_pickle=True)
    projection = torch.tensor(temp, dtype=torch.float).to(device)
    
    temp = np.load('mapping/layer-8/europarl-v7.%s-%s.%s.GBDD' % (src, tgt, args.layer), allow_pickle=True)
    bias = torch.tensor(temp, dtype=torch.float).to(device)

    data = pd.read_csv(os.path.join('WMT17', 'testset', path), sep='\t') 
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
    
    xmoverscores = scorer.compute_xmoverscore(args.alignment, projection, bias, source, translations, ngram=args.ngram, \
                                              layer=args.layer, dropout_rate=args.dropout_rate, bs=args.batch_size)

    lm_scores = scorer.compute_perplexity(translations, bs=1)
    
    scores = metric_combination(xmoverscores, lm_scores, [1, 0.1])
    
    print('\r\nlp:{} xmovescore:{} xmoverscore+lm:{}'.format(lp, 
                                      pearson(human_score, xmoverscores),
                                      pearson(human_score, scores)))
