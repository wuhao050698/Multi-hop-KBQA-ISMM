
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

1. How to compile
No Need to compile with this python file.

2. How to Execute
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

3.The description of each source file
3.1. train_embeddings: the file include all the python script to training the embedding for question and Knowledge
3.2. KGQA: the file include all the python script run to train the models (Include LSTM,RoBERTa)
3.3. kge: A very helpful library and we suggest that you train embeddings through it since it supports sparse embeddings + shared negative sampling to speed up learning for large KGs like Freebase.
3.4. data(Need to download and Unzip): data all the data file (KG dataset and Question Answering DataSet)
3.5  pretrained_modelsa(Need to download and Unzip): inclued all the pretrain ComplEx embedding models




4. An example to show how to run the program
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

5. Operating System tested our program : Windows
