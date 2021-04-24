# Research of Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model

This is the code for the MSBD5002 project.

# [detail for ISMM](ISMM/README.md)
# [detail for EmbedKGQA](MLRC2020-EmbedKGQA/readme.md)


# Iterative Sequence Matching Model
This is the code for paper:\
**[Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=5939&context=sis_research)**\
Yunshi Lan, Shuohang Wang, Jing Jiang\
Will appear at [ICDM 2019](http://icdm2019.bigke.org/) .

If you find this code useful in your research, please cite
>@inproceedings{lan:icdm19,\
>title={Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model},\
>author={Lan, Yunshi and Wang, Shuohang and Jiang, Jing},\
>booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},\
>year={2019}\
>}

## **Setups** 
All code was developed and tested in the following environment. 
- Ubuntu 16.04
- Python 3.7.1
- Pytorch 1.1.0

Download the code and data:
```
git clone https://github.com/lanyunshi/Multi-hopQA.git
```
## How to Execute
Open the ```ISMM.ipynb``` file, it contains the instructions needed to run the code.



## **the description of each source file**
 The data set and source code can be obtained from [here](https://drive.google.com/drive/folders/1P0KnElxzxTufJLqedriYbMOL4jCbjiwS).
### Data Description

The MetaQA dataset consist of several files.:
1. **qa_train.txt, qa_dev.txt, qa_test.txt**. These are raw data of training, validation and test set. Each line is string of a question followed by the answer.
2. **qa_train_qtype.txt, qa_dev_qtype.txt, qa_test_qtype.txt**. These are data of training, validation and test set after preprocessing. Each line is a chain, the first and last word is entities, and relations in the middle.
3. **kb.txt, kb_half.txt**. Raw data of knowledge graph, each line is string of a (entity, relation, entity) triple.

### Source code Description
1. **Multi-hopKBQA_runner.py**. Main function.
2. **advanced_layers.py**. Layers for calculating scores.
3. **models.py**. The main architecture of the model.
4. **util.py**. Some function including reading data, printing result and so on.
5. **options.py**. Some options to set arguement, to modify parameter of the model.

## **An example to show how to run the program**
```
python code/Multi-hopKBQA_runner.py --task 1 # Run pre-trained datasets: {1, 2, 3}
```
A full list of commands can be found in ```code/options.py```. The training script has a number of command-line flags that you can use to configure the model architecture, hyper-parameters and input/output settings. You can change the saved arguments in ```code/Multi-hopKBQA_runner.py``` .

## **Running a New Model**
```
python code/Multi-hopKBQA_runner.py --task 0
```
Task 0 is set to train your own model. The data is pre-processed and the model is initialized randomly. To train a new model, a new folder will be generated in the ```trained_model/```. A general good argument setting is:
- learning_rate : 0.0001
- hidden_dimension : 200 
- dropout : 0.0
- max_epochs : 20
- threshold : 0.5
- max_hop : 3
- top : 3

* Note that in our project, the code to get parts of  experimental results is obtained by modifying the default parameters in the ```options.py```, so our notebook file may not contain all the parameters.

## which System tested our program : Colab(linux)





# EmbedKGQA
This is the code for ACL 2020 paper [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf)

UPDATE: Code for relation matching has been added. Please see the [Readme](KGQA/RoBERTa/README.md) for details on how to use it.
![](model.png)
## Requirements
- Python >= 3.7.5, pip
- zip, unzip
- Pytorch version [1.3.0a0+24ae9b5](https://github.com/pytorch/pytorch/tree/24ae9b504094937fbc7c24012fbe5c601e024bcd).
- [transformers](https://github.com/huggingface/transformers)
- linux/Colab, use GPU

# 1. How to compile
No Need to compile with this python file.

# 2. How to Execute
In order to run the code, first download data.zip and pretrained_model.zip from [here](https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing). Unzip these files in the main directory.

## MetaQA

Change to directory ./KGQA/LSTM. Following is an example command to run the QA training code

```
python3 main.py --mode train --relation_dim 200 --hidden_dim 256 \
--gpu 2 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2  --scoredrop 0.2 \
--decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half
```

## WebQuestionsSP

Change to directory ./KGQA/RoBERTa. Following is an example command to run the QA training code
```
python3 main.py --mode train --relation_dim 200 --do_batch_norm 1 \
--gpu 2 --freeze 1 --batch_size 16 --validate_every 10 --hops webqsp_half --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 \
--decay 1.0 --model ComplEx --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile half_fbwq
```
Note: This will run the code in vanilla setting without relation matching, relation matching will have to be done separately.

Also, please not that this implementation uses embeddings created through libkge (https://github.com/uma-pi1/kge). This is a very helpful library and I would suggest that you train embeddings through it since it supports sparse embeddings + shared negative sampling to speed up learning for large KGs like Freebase.

# Dataset creation

## MetaQA

### KG dataset

There are 2 datasets: MetaQA_full and MetaQA_half. Full dataset contains the original kb.txt as train.txt with duplicate triples removed. Half contains only 50% of the triples (randomly selected without replacement). 

There are some lines like 'entity NOOP entity' in the train.txt for half dataset. This is because when removing the triples, all triples for that entity were removed, hence any KG embedding implementation would not find any embedding vector for them using the train.txt file. By including such 'NOOP' triples we are not including any additional information regarding them from the KG, it is there just so that we can directly use any embedding implementation to generate some random vector for them.

### QA Dataset

There are 5 files for each dataset (1, 2 and 3 hop)
- qa_train_{n}hop_train.txt
- qa_train_{n}hop_train_half.txt
- qa_train_{n}hop_train_old.txt
- qa_dev_{n}hop.txt
- qa_test_{n}hop.txt

Out of these, qa_dev, qa_test and qa_train_{n}hop_old are exactly the same as the MetaQA original dev, test and train files respectively.

For qa_train_{n}hop_train and qa_train_{n}hop_train_half, we have added triple (h, r, t) in the form of (head entity, question, answer). This is to prevent the model from 'forgetting' the entity embeddings when it is training the QA model using the QA dataset. qa_train.txt contains all triples, while qa_train_half.txt contains only triples from MetaQA_half.

## WebQuestionsSP

### KG dataset

There are 2 datasets: fbwq_full and fbwq_half

Creating fbwq_full: We restrict the KB to be a subset of Freebase which contains all facts that are within 2-hops of any entity mentioned in the questions of WebQuestionsSP. We further prune it to contain only those relations that are mentioned in the dataset. This smaller KB has 1.8 million entities and 5.7 million triples. 

Creating fbwq_half: We randomly sample 50% of the edges from fbwq_full.

### QA Dataset

Same as the original WebQuestionsSP QA dataset.

# 3.The description of each source file

3.1. train_embeddings: the file include all the python script to training the embedding for question and Knowledge

3.2. KGQA: the file include all the python script run to train the models (Include LSTM,RoBERTa)

3.3. kge: A very helpful library and we suggest that you train embeddings through it since it supports sparse embeddings + shared negative sampling to speed up learning for large KGs like Freebase.

3.4. data(Need to download and Unzip): data all the data file (KG dataset and Question Answering DataSet)

3.5  pretrained_modelsa(Need to download and Unzip): inclued all the pretrain ComplEx embedding models




# 4. An example to show how to run the program
For exmpale, run the EmbedKGQA on 2 hops questions from MetaQA dataset with KG-50.

Step1:Download data.zip and pretrained_model.zip from [here](https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing). Unzip these files in the main directory.

Step2:Change to directory ./KGQA/LSTM. Run following command to run the QA training code.

```
python3 main.py --mode train --relation_dim 200 --hidden_dim 256 \
--gpu 0 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2  --scoredrop 0.2 \
--decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half
```

Two Important Parameters:

Hops Question Type: 
--hops 1
--hops 2
--hops 3

Knowledge Graph Type:
--kg_type Full
--kg_type half

# 5. Operating System tested our program : Windows
