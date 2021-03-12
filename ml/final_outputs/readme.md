# ML-CB: Final Outputs

- [Introduction](#introduction)
- [Example](#example)

## Introduction

This folder holds the output from running `main.py` or `ml-cb.py` or `cnn3.py`. 

> Please note, this build is setup to replicate the results in the paper, specifically Table 4; however, due to variations in examples included in testing (i.e., random downsampling from the majority class), the results are expected to differ to some degree. 

### External file structure (naming)

`<model>--<corpus>.csv`

- **model**: {SVM, BoW, Embedding}
- **corpus**:  {plaintext, jsNice}

### Internal file structure (columns)

- metrics reported: accuracy, f1, precision, recall
- 20 columns in total
  - [*Original Scrape*] 1-4: results on test-train split 
  - [*Test Suite*] 5-8: train on selected corpus, test on test suite plaintext
  - [*Test Suite*] 9-12: train on selected corpus, test on test suite plaintext-then-jsNiceified
  - [*Adversarial Perspective*] 13-16: train on selected corpus, test on test suite plaintext-obfuscated
  - [*Adversarial Perspective*] 17-20: train on selected corpus, test on test suite plaintext-obfuscated-jsNiceified 
- column naming
  - `_original` &mdash; test-train split with plaintext corpus *OR* test-train split with jsNice corpus
  - `_program_plain`  &mdash; plaintext test suite corpus
  - `_program_jsnice_v2` &mdash; jsNice test suite corpus
  - `_program_plain_obfuscated` &mdash; plaintext-obfuscated test suite corpus
  - `_program_plain_obfuscated_jsnice_v2` &mdash; plaintext-obfuscated-jsNiceified test suite corpus



## Example

*using `main.py` on `embedding` model with `jsnice` corpus* 

**external structure**

```
embedding--jsnice.csv
```

**internal structure**

- columns 1-4: test train split, training with jsNice corpus, testing with jsNice corpus (split)
- columns 5-8: train jsNice corpus, test with test suite plaintext data (non-jsNiceified)
- columns 9-12: train jsNice corpus, test with test suite jsNiceified plaintext data
- columns 13-16: train jsNice corpus, test with test suite plaintext-obfuscated da
- columns 17-20: train jsNice corpus, test with test suite plaintext-obfuscated-jsniceifed data 

**results interpretation**

*each of the n rows (i.e., each run) would be averaged*