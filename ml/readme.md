# ML-CB: Install

- [disclaimers](#disclaimers)
- [ubuntu](#ubuntu)
  * [basic packaages](#basic-packaages)
  * [anaconda](#anaconda)
  * [repository](#repository)
  * [data](#data)
- [macos](#macos)
  * [anaconda](#anaconda-1)
  * [repository](#repository-1)
  * [data](#data-1)
  * [Sanity Check](#sanity-check)
- [Running ML-CB](#running-ml-cb)

## disclaimers

- a computing instance geared toward machine learning, with high ram and sufficient space, is necessary to run these programs. Without sufficient RAM and disk space, the program (it is currently not optimized) will likely be killed. 

  - preferred test method: Google's Compute Cloud (VM instance) 
    - set a new VM to 8 CPUs (N2) with 350 GB RAM and 100GB disk space (us-central1-a), then follow the ubuntu steps for install 

- requires **python 3.7** ==> tested with python 3.7.9 (3.7.6 may also work)

- this code (and subsequently the install instructions) is setup to work only on CPUs; GPU would increase performance, but is not tested in this release

  

---

<ol>
  <li><p style="color: red">Install {Ubuntu or macOS}</p></li>
  <li><p style="color: red">Sanity Check</p></li>
  <li><p style="color: red">Run ML-CB</p></li>
</ol> 

---



## ubuntu 

> tested with (Debian GNU/Linux 10 (buster))

### basic packaages

```bash
sudo apt-get install unzip -y
sudo apt-get install wget -y
sudo apt-get install git -y
```

### anaconda

1. install anaconda 
     1. recommended: https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04
     
        1. ```bash
           cd /tmp
           wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
           bash Anaconda3-2020.02-Linux-x86_64.sh
           source ~/.bashrc
           conda info
           ```
     
     2. should open a (base) conda environment (use the `/tmp` folder creation and the `source ~/.bashrc` in the end)
2. `conda install -c fastai -c pytorch -c anaconda fastai gh anaconda`
   
   1. *☝️ this command may take a few minutes to finish* 
  3. downgrade fastai for CNN: `pip install fastai==1.0.61`
4. `conda install tensorflow-mkl`
   
   1. *☝️ this command may take a few minutes to finish*

### repository

```bash
# cd into a directory location you prefer
git clone https://github.com/SP2-MC2/ML-CB.git
cd ML-CB
```

### data

1. get text corpora

   1. ```bash
      cd ml
      wget https://osf.io/ep7d5/download
      unzip download 
      rm -r download
      cd data/TEXT
      ```

2. get embedding pre-set weights (GloVe)

   1. ```bash
      wget https://osf.io/vshgb/download
      unzip download
      rm -r download
      ```

3. head back to central repository 

   1. ```bash
      cd ../..
      ```



---



## macos

> tested with  10.15 (11.2.2)

### anaconda

1. install anaconda
   1. a full list of options -- https://repo.anaconda.com/archive/
      1. recommended direct GUI download (**this is all you need**) : [Anaconda3-2020.02-MacOSX-x86_64.pkg](https://repo.anaconda.com/archive/Anaconda3-2020.02-MacOSX-x86_64.pkg)
2. use anaconda navigator to open a root (base) terminal (hit the play button in the environments tab)
   1. optionally, use any conda environment, just work within that environment
3. `conda install -c pytorch -c fastai fastai` 
   1. *☝️ this command may take a few minutes to finish*
  4. downgrade fastai for CNN: `pip install fastai==1.0.61`
5. `conda install tensorflow-mkl`
   1. *☝️ this command may take a few minutes to finish*

### repository

```bash
# cd into a directory you prefer 
git clone https://github.com/SP2-MC2/ML-CB.git
cd ML-CB
conda install -c anaconda wget 
```

### data

1. get text corpora

   1. ```bash
      cd ml
      wget https://osf.io/ep7d5/download
      unzip download 
      rm -r download
      cd data/TEXT
      ```

2. get embedding pre-set weights  (GloVe)

   1. ```bash
      wget https://osf.io/vshgb/download
      unzip download
      rm -r download
      ```

3. head back to central repository 

   1. ```bash
      cd ../..
      ```



---



### Sanity Check 

Your file structure should look like this

```
.
├── cnn3.py  // runs image-based ML
├── data
│   ├── CNN
│   │   └── -released  // data for CNN 
│   │       └── split
│   │           ├── test
│   │           │   ├── false
│   │           │   └── true
│   │           ├── test_suite
│   │           │   ├── false
│   │           │   └── true
│   │           └── train
│   │               ├── false
│   │               └── true
│   └── TEXT  // data for text-based ML
│       ├── glove.42B.300d.txt
│       ├── jsnice_examples_v2.csv
│       ├── plain_examples.csv
│       └── test_suite.csv
├── final_outputs  // *** results stored here ***
│   └── README.txt
├── main.py  // wrapper for ml-cb.py
├── ml-cb.py  // runs text-based ML
└── readme.md
```



## Running ML-CB

> This build is setup to replicate the results in the paper, specifically Table 4. The text-based ML (`ml-cb.py`) will, by default, store the results of each run in the final_outputs folder as a .csv file.
>
> 
>
> Additionally, the models are not automatically saved. A function is available (`save_model`), but is currently not called in this release. 



To run the text-based machine learning with the wrapper

```python
python3 main.py 
```

To run the image-based machine learning 

```python
python3 cnn3.py
```

You can also run ML-CB directly

```
python3 ml-cb.py --model embedding --corpus jsnice
```

- **model**: which machine learning model to use (SVM, Bag of Words, Embedding) 
  - `python3 ml-cb.py --model svm --corpus jsnice`
- **corpus**: which text base to use (plaintext website source code or jsNiceified website source code)
  - `python3 ml-cb.py --model embedding --corpus plaintext`



> to reproduce results from the paper, edit main.py to run for 5 times on each model and each corpus, then average the results from the five runs



---

To remove conda, follow: https://docs.anaconda.com/anaconda/install/uninstall/

