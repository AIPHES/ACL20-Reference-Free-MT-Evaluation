# ACL20-Reference-Free-MT-Evaluation
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) 

XMoverScore is a metric for reference-free MT evaluation, described in the paper [On the Limitations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation](https://www.aclweb.org/anthology/2020.acl-main.151.pdf) (ACL 2020).

XMoverScore is a cross-lingual extension of [MoverScore](https://github.com/AIPHES/emnlp19-moverscore) for Machine Translation Evaluation. We have also released a similar metric for summarization, called [SUPERT](https://github.com/yg211/acl20-ref-free-eval).


**CURRENT VERSION:**
* We provide a reference-free metric coupled with re-aligned multilingual BERT and a target-side LM (GPT-2).
* We provide the metrics evaluation on the WMT17, WMT18 and WMT19 datasets.
* The remapping weights are released, with 11 supported languages involving German, Chinese, Czech, Latvian, Finnish, Russian, Turkish, Gujarati, Kazakh, Lithuanian and Estonian. 
* We provide [manual annotations](https://docs.google.com/spreadsheets/d/1kLGk66TgUzSRftm_7Xir5ehhnU1AqD5XR8pM44mDSTI/edit?usp=sharing) of cross-lingual (German-English and Russian-English) DA scores for source-translation pairs.
* Since our metric uses BERT and GPT-2, a GPU is necessary.
* Note that the current version uses [normalized language model scores](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation/commit/cc4f4fb8f529a1a4b12bc303c4ca98bbc97069b6). For reproducibility, replacing them with negative log-likelihoods is necessary. 

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

XMoverScore is Apache-licensed, as found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

## References


```
@inproceedings{zhao-etal-2020-limitations,
    title = "On the Limitations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation",
    author = "Zhao, Wei and Glava{\v{s}}, Goran and Peyrard, Maxime and Gao, Yang and West, Robert and Eger, Steffen",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.151",
    pages = "1656--1671"
}
```

```
@inproceedings{gao-etal-2020-supert,
    title = "{SUPERT}: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization",
    author = "Gao, Yang  and Zhao, Wei and Eger, Steffen",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.124",
    pages = "1347--1354"   
}
```

```
@inproceedings{zhao-etal-2019-moverscore,
    title = "{M}over{S}core: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance",
    author = "Zhao, Wei and Peyrard, Maxime and Liu, Fei and Gao, Yang and Meyer, Christian M. and Eger, Steffen",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1053",
    doi = "10.18653/v1/D19-1053",
    pages = "563--578",
}
```

