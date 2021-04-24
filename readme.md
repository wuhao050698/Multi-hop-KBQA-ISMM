# Research of Multi-hop Knowledge Base Question Answering with an Iterative Sequence Matching Model

This is the code for the MSBD5002 project.

## Requirements
- Python >= 3.7.5, pip
- zip, unzip
- Pytorch version [1.3.0a0+24ae9b5](https://github.com/pytorch/pytorch/tree/24ae9b504094937fbc7c24012fbe5c601e024bcd).
- [transformers](https://github.com/huggingface/transformers)
- linux/Colab, use GPU

## Iterative Sequence Matching Model
 
## EmbedKGQA
MLRC2020-EmbedKGQA/: the code for the EmbedKGQA

### Instructions
In order to run the code, first download data.zip and pretrained_model.zip from [here](https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq-0KcjSd3ujxc?usp=sharing). Unzip these files in the main directory.

### MetaQA

Change to directory ./KGQA/LSTM. Following is an example command to run the QA training code

```
python3 main.py --mode train --relation_dim 200 --hidden_dim 256 \
--gpu 2 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2  --scoredrop 0.2 \
--decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half
```

### WebQuestionsSP

Change to directory ./KGQA/RoBERTa. Following is an example command to run the QA training code
```
python3 main.py --mode train --relation_dim 200 --do_batch_norm 1 \
--gpu 2 --freeze 1 --batch_size 16 --validate_every 10 --hops webqsp_half --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 \
--decay 1.0 --model ComplEx --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile half_fbwq
```
