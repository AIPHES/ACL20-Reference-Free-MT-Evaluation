import pandas as pd
import numpy as np
import json
import tqdm
import os 
    
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
parser.add_argument('--do_lower_case', type=bool, default=False)
parser.add_argument('--language_model', type=str, default='gpt2')
parser.add_argument('--mapping', type=str, default='CLP', help='CLP or UMD')    
parser.add_argument('--dataset', type=str, default='WMT18', help='WMT17, WMT18, WMT19')  

args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))

from scorer import XMOVERScorer

scorer = XMOVERScorer(args.model_name, args.language_model, args.do_lower_case)

 
import torch
import truecase
from mosestokenizer import MosesDetokenizer
from mt_utils import (find_corpus, 
                      load_data, 
                      load_metadata, 
                      print_sys_level_correlation, 
                      print_seg_level_correlation,
                      df_append)
               
dataset = find_corpus(args.dataset)           

wmt_xmoverscores = []
for pair in dataset.items():
    reference_path, lp = pair
    
    src, tgt = lp.split('-')
    references = load_data(os.path.join(args.dataset, 'references/', reference_path))
    
    source_path = reference_path.replace('ref', 'src')
    source_path = source_path.split('.')[0] + '.' + src  
    source = load_data(os.path.join(args.dataset, 'source', source_path))
    
    all_meta_data = load_metadata(os.path.join(args.dataset, 'system-outputs', lp))
    
    with MosesDetokenizer(src) as detokenize:        
        source = [detokenize(s.split(' ')) for s in source]         
    with MosesDetokenizer(tgt) as detokenize:                
        references = [detokenize(s.split(' ')) for s in references]
      
    device='cuda:0'
    temp = np.load('mapping/layer-12/europarl-v7.' + src + '-' + tgt + '.2k.12.BAM', allow_pickle=True)
    projection = torch.tensor(temp, dtype=torch.float).to(device)
    
    temp = np.load('mapping/layer-12/europarl-v7.' + src + '-' + tgt + '.2k.12.GBDD', allow_pickle=True)
    bias = torch.tensor(temp, dtype=torch.float).to(device)
    
    for i in tqdm.tqdm(range(len(all_meta_data))):
        path, testset, lp, system = all_meta_data[i]
        
        translations = load_data(path)        
        num_samples = len(references)
        df_system = pd.DataFrame(columns=('metric', 'lp', 'testset', 'system', 'sid', 'score'))
        
        with MosesDetokenizer(tgt) as detokenize:                    
            translations = [detokenize(s.split(' ')) for s in translations]
        
        translations = [truecase.get_true_case(s) for s in translations]
                
        xmoverscores = scorer.compute_xmoverscore(args.mapping, projection, bias, source, translations, ngram=1, bs=64)
    
        wmt_xmoverscores.append(df_append('xmoverscore', num_samples, lp, testset, system, xmoverscores)) 

print_sys_level_correlation('xmoverscore', wmt_xmoverscores, list(dataset.values()), os.path.join(args.dataset, 'DA-syslevel.csv'))

if args.dataset not in ['WMT17']:
    print_seg_level_correlation('xmoverscore', wmt_xmoverscores, list(dataset.values()), os.path.join(args.dataset, 'RR-seglevel.csv'))


