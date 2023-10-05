# Numerical Question Generation in Science

## Steps to Run Code
- The code is present in the /code/diffuSeq_qg.ipynb notebook.
- Run all the cells serially
  - Update train.sh with required parameters
  - `python -m torch.distributed.launch --nproc_per_node=1 --master_port=12233 --use_env run_train.py \
      --diff_steps 2000 \
      --lr 0.0001 \
      --learning_steps 40000 \
      --save_interval 2000 \
      --seed 102 \
      --noise_schedule sqrt \
      --hidden_dim 128 \
      --resume_checkpoint /content/gdrive/MyDrive/BTP_1/weights/openQA_40000 \
      --bsz 32 \
      --dataset qg \
      --data_dir /content/gdrive/MyDrive/BTP_1/dataset/electric_charge \
      --vocab bert \
      --seq_len 128 \
      --schedule_sampler lossaware \
      --notes qg`
  - `bash train.sh` - Used to train the model on the required datasets
  - `bash /content/DiffuSeq/scripts/run_decode.sh` - Used to apply the trained weights on test.jsonl file to get results/questions generated from the model
  - `python eval_seq2seq.py --folder FOLDER_NAME` - Used to score the generated results


## Objective
The main objective of this project is to develop a model that can automatically generate numerical questions in science based on a given theory. To achieve this goal, different approaches will be explored to model the relationships between the context and questions in order to improve the diversity of the generated questions. The quality and diversity of the questions generated will be assessed through various metrics and evaluation techniques

## Dataset
- OpenQA
- Forward Mapping
- Forward and Backward Mapping

Some of the datasets couldn't be uploaded due to their large size.
All the datasets and results are uploaded in the below link.

## Results

All the datasets and results are uploaded in the below link.
https://drive.google.com/drive/folders/11ZVFibhebYtb_mGHcaluJhMjBAA13QIz?usp=sharing
