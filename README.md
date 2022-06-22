# Paraphrase Generator
![version](https://img.shields.io/badge/version-2.1.0-blueviolet)
![python](https://img.shields.io/badge/python-3.7.7-brightgreen)

Paraphrase Generator in Pytorch using Seq2Seq GRU-LSTM Model with Shared Pair-Wise Discriminator and Lexical Modification using NLTK, Spacy and Pattern3. 
#### Seq2Seq Model
The motivation for the model is derived from the paper [Learning Semantic Sentence Embeddings using Pair-wise Discriminator] (https://www.aclweb.org/anthology/C18-1230.pdf)

#### Lexical Modification Model
This model facilitates paraphrase modification by synonym replacement, antonym replacement with elimination of apropos negation and tense rectifier (tense identification and transformation), while retaining the semantics of the original sentence.

## Requirements and Installation

##### Use Google Colab or a Code Editor IDE

1. Clone this repository (and create a virtual environment)
```
git clone https://github.com/smit25/Paraphrase-Generation
cd Paraphrase-Generation
```

2. Download venv for creating a virtual environment
```
pip install --user virtualenv
pip install requests

## For Windows
py -m venv env

## For MAC and Linux
python3 -m venv env
```
3. After that for logging you need to install [tensorboardX](https://github.com/lanpa/tensorboardX).
```
pip install tensorboardX
```

### Dataset
You can directly use following files downloading them into `data` folder or by following the process shown below it.
##### Quora Duplicate Questions Dataset
Download all the data files from here.
- [quora_data_prepro.h5](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_data_prepro.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_train.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_val.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_test.json](https://figshare.com/s/5463afb24cba05629cdf)

##### Google PAWS Dataset
- [PAWS_train_final.json](https://github.com/smit25/Paraphrase/blob/master/data/PAWS_train_final.json)
- [PAWS_train_swap.json](https://github.com/smit25/Paraphrase/blob/master/data/PAWS_train_swap.json)

##### Download Dataset
If you want to train from scratch continue reading, or if you just want to evaluate using a pretrained model then head over to `data` section and download the data files and run `train.py` to evaluate on the pretrained model.

For preprocessing on the original tsv file, head to 'preprocessing' folder and run:
``` 
cd preprocessing
# Quora Dataset
python split.py 
# PAWS Dataset
python PAWS_split.py
```

**Note** The above command generates json files for 100K question pairs for train, 20k question pairs for validation and 20K question pairs for Test set.
If you want to change the dataset length, then make some minor changes in the 'split.py'. After this step, it will generate the files under the `data` folder. `quora_raw_train.json`, `quora_raw_val.json` and `quora_raw_test.json`.

##### Preprocessing on the Quora Dataset
```
$ python preprocess.py --input_train_json ../data/quora_raw_train.json --input_test_json ../data/quora_raw_test.json 
# OR JUST
$ python preprocess.py
```
This will generate two files in `data/` folder, `quora_data_prepro.h5` and `quora_data_prepro.json`.


### Training
```
$ python train.py --n_epoch <number of epochs>
```
There are other arguments also for you to experiment like `--batch_size`, `--learning_rate`, `--drop_prob_lm`, and so on.

### Save and log
First you have to make empty directories `save`, `samples`, and `logs`.  
For each training there will be a directory having unique name in `save`. Saved model will be a `.tar` file. Each model will be saved as `<epoch number>` in that directory.

In `samples` directory with same unique name as above the directory contains a `.txt` file for each epoch as `<epoch number>_train.txt` or `<epoch number>_val.txt` having generated paraphrases by model at the end of that epoch on validation data set.

Logs for training and evaluation is stored in `logs` directory which you can see using `tensorboard` by running following command.
```
tensorboard --logdir <path of logs directory>
```
This command will tell you where you can see your logs on browser, commonly it is `localhost:6006` but you can change it using `--port` argument in above command.

### Modification Introduced
1. Proper Noun Segregation: All proper nouns (pos_tag = ‘PPN’) in the sentence are stored locally and replaced with ‘UNK’. After the paraphrase for the sentence is generated, the ‘UNK’ are replaced with the original words.
2. Expansion of the Bag of Words: Originally, NLTK corpus was used to make the BOW, but limitations of Google Colab (tensor memory issue) made it redundant, hence words from Project Gutenberg were used to expand the BOW.


### Results
Following are the results for 100k quora question pairs dataset.

Name of model | Bleu_1 | Bleu_2 | ROUGE_L | CIDEr |
---|--|--|--|--|
Orig|0.5150|0.3491|0.5263|1.5922|
Modified|0.4561|0.2933|0.4820|1.0274|

### Testing or Running the Model
You can test the model by entering the sentence in 'generate_parahrase.py' and running the code:
```
python generate_paraphrase.py
```
The Seq2Seq trained model and the lexical modifiers are called simultaneously and the paraprase in procured as the output.

**Note** To apply lexical modification only, change the code in 'generate_paraphrase.py' and directly put the input sentence in the modifier object instances.

##### Examples
Some examples of paraphrase generated during validation.

Original Sentence | Paraphrase Generated
---|---|
How should one start preparations for CAT|How do I start learning for CAT
What are the safety precautions on handling shotguns proposed by the NRA in Florida|What are the safety precautions on handling shotguns proposed by the NRA in Idaho
Why do people ask questions on Quora that can easily be answered by Google|Why do people ask questions on Quora that could be be found by Google
What will be the implications of banning 500 and 1000 rupees currency notes on Indian economy|What will the safety effects of demonetization of 500 and 1000 rupee notes

### References
The references used throughout the project are mentioned in 'misc/references.py'.
```
@inproceedings{patro2018learning,
  title={Learning Semantic Sentence Embeddings using Sequential Pair-wise Discriminator},
  author={Patro, Badri Narayana and Kurmi, Vinod Kumar and Kumar, Sandeep and Namboodiri, Vinay},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics},
  pages={2715--2729},
  year={2018}
}
```
### Contribution
* Smit Patel
* The original authors of the paper (reference)

For any queries, feel free to contact me at smitu3435@gmail.com
