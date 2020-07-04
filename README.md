# ACL20-Reference-Free-MT-Evaluation
XMoverScore, is a metric for reference-free MT evaluation, described in the paper [On the Limitations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation] (https://www.aclweb.org/anthology/2020.acl-main.151.pdf) (ACL 2020)

**CURRENT VERSION:**
* We provide a reference-free metric coupled with re-aligned multilingual BERT and a target-side LM (GPT-2).
* We provide the metrics evaluation on the WMT17, WMT18 and WMT19 benchmark datasets.
* The remapping weights are released, with 11 supported languages involving German, Chinese, Czech, Latvian, Finnish, Russian, Turkish, Gujarati, Kazakh, Lithuanian and Estonian. 
* We provide manual annotations of cross-lingual (German-English and Russian-English) DA scores for source-translation pairs.
* Since our metric uses BERT and GPT-2, a GPU is necessary.

## Dependencies
* Python 3.6
* [PyTorch](http://pytorch.org/), tested with 1.3.1
* [NumPy](http://www.numpy.org/), tested with 1.18.4
* [Pyemd](https://github.com/wmayner/pyemd), fast earth mover distance, tested with 0.5.1
* [Transformers](https://github.com/huggingface/transformers), multilingual BERT and GPT-2, tested with 2.7.0
* [Mosestokenizer] tokenization from the Moses encoder, tested with 1.0.0

## Usage

### Python Function
We provide a python object `XMOVERScorer` which caches multilingual BERT and a target-side LM and wrapps the implementations of our metric. Check our [demo](demo.py) on the WMT17 testset to see its usage. Please refer to [`score_utils.py`](score_utils.py) for the implementation details.

### Reproducing Results
We provide [`main.py`](main.py) which could reproduce the results, i.e., the system-level and seg-level correlations of metrics and human judgments (DA scores), reported in the paper.

## License

XMoverScore is BSD-licensed, as found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

## References

[1] Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, Steffen Eger,
    [*On the Limitations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation*](https://www.aclweb.org/anthology/2020.acl-main.151.pdf),
    ACL 2020


[2] Yang Gao, Wei Zhao, Steffen Eger
    [*SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization*](http://aclweb.org/anthology/P18-2037)
    ACL 2020

[3] Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger,
    [*MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance*](http://www.lrec-conf.org/proceedings/lrec2018/pdf/658.pdf),
    EMNLP 2019

