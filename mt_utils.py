import glob
from io import StringIO
import pandas as pd
import os
from scipy.stats import pearsonr

def find_corpus(name):
    
    WMT2017 = dict({
            "newstest2017-csen-ref.en": "cs-en",
            "newstest2017-deen-ref.en": "de-en",
            "newstest2017-fien-ref.en": "fi-en",
            "newstest2017-lven-ref.en": "lv-en",
            "newstest2017-ruen-ref.en": "ru-en",
            "newstest2017-tren-ref.en": "tr-en",
            "newstest2017-zhen-ref.en": "zh-en"
            })   
             
    WMT2018 = dict({
            "newstest2018-csen-ref.en": "cs-en",
            "newstest2018-deen-ref.en": "de-en",
            "newstest2018-eten-ref.en": "et-en",        
            "newstest2018-fien-ref.en": "fi-en",
            "newstest2018-ruen-ref.en": "ru-en",
            "newstest2018-tren-ref.en": "tr-en",
            "newstest2018-zhen-ref.en": "zh-en",
            })    
    
    WMT2019 = dict({
            "newstest2019-deen-ref.en": "de-en",
            "newstest2019-fien-ref.en": "fi-en",
            "newstest2019-guen-ref.en": "gu-en",
            "newstest2019-kken-ref.en": "kk-en",
            "newstest2019-lten-ref.en": "lt-en",
            "newstest2019-ruen-ref.en": "ru-en",
            "newstest2019-zhen-ref.en": "zh-en",
            }) 
    
    if name == 'WMT17':
        dataset = WMT2017
    if name == 'WMT18':
        dataset = WMT2018
    if name == 'WMT19':
        dataset = WMT2019        
    return dataset

def load_data(path):
    lines = []
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            lines.append(l)
    return lines

def load_metadata(lp):
    files_path = []
    for root, directories, files in os.walk(lp):
        for file in files:
            if '.hybrid' not in file:
                raw = file.split('.')
                testset = raw[0]
                lp = raw[-1]
                system = '.'.join(raw[1:-1])
                files_path.append((os.path.join(root, file), testset, lp, system))
    return files_path

def df_append(metric, num_samples, lp, testset, system, score):
    return pd.DataFrame({'metric': [metric] * num_samples,
                         'lp': [lp] * num_samples,
                         'testset': [testset] * num_samples,
                         'system': [system] * num_samples,
                         'sid': [_ for _ in range(1, num_samples + 1)],
                         'score': score,
                        })
    
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    return str('{0:.{1}f}'.format(pearson_corr, 3))
    
def output_MT_sys_level_correlation(lp_set, eval_metric, f):
    submissions = ["%s.sys.score" % (eval_metric)]
    lines = [line.rstrip('\n') for line in open(f)]
    lines.pop(0)
    
    manual = {}
    
    for l in lines:
      l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")
    
      c = l.split()
    
      if len(c) != 3:
        print ("erorr in manual evaluation file")
        exit(1)
     
      lp = c[0]
      score = c[1]
      system = c[2] 
    
      if lp not in manual:
        manual[lp] = {}
      if system not in manual[lp]:
        manual[lp][system] = score
      
    missing = 0
    
    met_names = {}
    lms = {}
    lsm = {}
    
    for s in submissions:
      files = glob.glob(s)
      for f in files:
    
        lines = [line.rstrip('\n') for line in open(f)]
    
        for l in lines:
          l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")
                        
          if (l.find("hybrid")==-1) and (l.find("himl")==-1):
    
            c = l.split()
    
            if ((len(c) != 5) and len(c)!=7) and (len(c)!=9):
              missing = missing + 1
    
            else:
              metric, lp, data, system, score = c[0], c[1], c[2], c[3], c[4]
              
              if data not in ["newstest2017", "newstest2018"]:
                print ("error with data set for metric: "+l)
                exit(1)
    
              if lp not in lms:
                lms[lp] = {}
              if metric not in lms[lp]:
                lms[lp][metric] = {}
              if system not in lms[lp][metric]:
                lms[lp][metric][system] = score
              
              if lp not in lsm:
                lsm[lp] = {}
              if system not in lsm[lp]:
                lsm[lp][system] = {}
              if system not in lsm[lp][system]:
                lsm[lp][system][metric] = score
              
    for lp in manual:
      if lp not in lp_set: continue    
      for metric in lms[lp]:
        if sorted(lms[lp][metric])==sorted(manual[lp]):      
          if lp not in met_names:
            met_names[lp] = {}
          if metric not in met_names[lp]:
            met_names[lp][metric] = 1
    
        else:
          print ("systems mismatch "+lp+" "+metric)
          print (sorted(lms[lp][metric]))
          print (sorted(manual[lp]))
    res_str = ""
    for lp in manual:
      if lp not in lp_set: continue   
      l = lp.replace("-","")
      s = "LP SYSTEM HUMAN"
    
      for metric in sorted(met_names[lp]):
          s = s+" "+metric
    
      s = s+"\n"    
      for system in manual[lp]:
        s = s+lp+" "+system+" "+manual[lp][system]    
        for metric in sorted(met_names[lp]):    
          s = s +" "+lsm[lp][system][metric]
        s = s+"\n"
      results = pd.read_csv(StringIO(s), sep=" ")    
      res_str = res_str + eval_metric + '\t'+ lp +"\t" + pearson_and_spearman(results['HUMAN'], results[eval_metric]) +"\n"
    return pd.read_csv(StringIO(res_str), sep="\t", header=None)

import gzip
def output_MT_seg_level_correlation(lp_set, eval_metric, f):   
    submissions = eval_metric
    
    lines = [line.rstrip('\n') for line in open(f)]
    lines.pop(0)

    manual = {}

    for l in lines:
        l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")
        c = l.split()

        if len(c) != 5:
            print ("error in manual evaluation file")
            exit(1)

        lp = c[0]
        data = c[1]
        sid = c[2] 
        better = c[3]
        worse = c[4]

        if lp not in manual:
            manual[lp] = {}
        if sid not in manual[lp]:
            manual[lp][sid] = {}
        if better not in manual[lp][sid]:
            manual[lp][sid][better] = {}
        if worse not in manual[lp][sid][better]:
            manual[lp][sid][better][worse] = 1
    
    missing = 0
    met_names = {}
    metrics = {}
    
    for s in submissions:
      files = glob.glob(s)
      for f in files:
        lines = [str(line, encoding='utf-8') for line in gzip.open(f)]   
        for l in lines:
          l = l.replace("nmt-smt-hybrid","nmt-smt-hybr")                    
          if (l.find("hybrid")==-1) and (l.find("himl")==-1):
            c = l.split()

            if len(c) < 6:
              missing = missing + 1

            else: 
              metric = c[0]
              lp = c[1]
              data = c[2]
              system = c[3]
              sid = c[4]
              score = float(c[5]) 

              if data != "newstest2018":
                  continue

              if lp not in metrics:
                metrics[lp] = {}
              if metric not in metrics[lp]:
                metrics[lp][metric] = {}
              if sid not in metrics[lp][metric]:
                metrics[lp][metric][sid] = {}
              if system not in metrics[lp][metric][sid]:
                metrics[lp][metric][sid][system] = score

    for lp in manual:
        if lp not in lp_set: continue
        if lp not in metrics:
            print (lp+" not in metrics")
            exit(1)
      
        for metric in metrics[lp]:
            allthere = True
            for sid in manual[lp]:
                if not sid in metrics[lp][metric]:
                    allthere = False
                    print ("A) Missing "+lp+" "+metric+" "+sid+" no scores at all for this metric and sid")
                else:  
                   for s1 in manual[lp][sid]:
                       if not s1 in metrics[lp][metric][sid]:
                           allthere = False
                           print ("B) Missing "+lp+" "+metric+" "+sid+" "+s1+" no scores for this metric for sid and first  system")
                   for s2 in manual[lp][sid][s1]:
                       if not s2 in metrics[lp][metric][sid]:
                           allthere = False
                           print ("C) Missing "+lp+" "+metric+" "+sid+" "+s1+" "+s2+" no scores for this metric for sid and second system")
        
            if allthere:
                if lp not in met_names:
                    met_names[lp] = {}
                if metric not in met_names[lp]:
                    met_names[lp][metric] = 1

    res_str = ""
    for lp in manual:
        if lp not in lp_set: continue
        for metric in met_names[lp]:
            conc = 0
            disc = 0
            for sid in manual[lp]:    
                s = s+lp+" "+sid+" "
                for better in manual[lp][sid]:
                    for worse in manual[lp][sid][better]:
                        if better not in metrics[lp][metric][sid]:
                            print ("error "+lp+" "+metric+" "+better)                                                
                        score1 = metrics[lp][metric][sid][better]
                        score2 = metrics[lp][metric][sid][worse]
                        if score1 > score2:
                            conc = conc + 1
                        else:
                            disc = disc + 1
                            
            conc = float(conc)
            disc = float(disc)
            result = (conc-disc)/(conc+disc)
            res_str = res_str + metric + '\t'+ lp +"\t" + '{0:.{1}f}'.format(result, 3) +"\n"
    return pd.read_csv(StringIO(res_str), sep="\t", header=None)
            
def print_sys_level_correlation(metric, data, lp_set, f = "DA-syslevel.csv"):
    results = pd.concat(data, ignore_index=True)
    del results['sid']
    results = results.groupby(['metric', 'lp', 'testset','system']).mean()
    results = results.reset_index()
    results.to_csv(metric + '.sys.score', sep='\t', index=False, header=False)
    outputs = output_MT_sys_level_correlation(lp_set, metric, f)
    s = ' '.join(lp_set) + '\n'
    s = s + metric + ' '+' '.join([str('{0:.{1}f}'.format(outputs[(outputs[1]==lp)].values[0][-1], 3)) for lp in lp_set])
    print(s)
    
import shutil    
def print_seg_level_correlation(metric, data, lp_set, f= "RR-seglevel.csv"):
    results = pd.concat(data, ignore_index=True)
    results.to_csv(metric + '.seg.score', sep='\t', index=False, header=False)
    with open(metric + '.seg.score', 'rb') as f_in:
        with gzip.open(metric + '.seg.score.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    outputs = output_MT_seg_level_correlation(lp_set, [metric + '.seg.score.gz'], f)
    s = ' '.join(lp_set) + '\n'
    s = s + metric + ' '+' '.join([str('{0:.{1}f}'.format(outputs[(outputs[1]==lp)].values[0][-1], 3)) for lp in lp_set])
    print(s)
    