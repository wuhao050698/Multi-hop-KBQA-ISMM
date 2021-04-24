# **Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model**
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