This project implements a sequential recommendation model based on BERT4Rec to predict next-item interactions from user behavior sequences. Below is a detailed introduction to the project modules.

## File Structure
├── train_ratio.py              # Model training file that outputs metric changes during training and evaluates the model on the test set at the end
├── model1.py                   # Model training using HuggingFace’s BERT model
├── model2.py                   # Custom model architecture implemented from scratch
├── maskdata.py                 # Script for randomly masking the training set
├── evaluation_ratio.py         # Function to compute NDCG@10 and Recall@10 using a rolling window evaluation method
├── data_load_ratio.py          # Data loading script that splits the data into 70/15/15 train/validation/test sets
└── README.md                   # Project description file

## Run the Code
Run train_ratio.py and it will start the model training, output the indicator changes of the validation set during the training process, and finally output the indicator evaluation on the test set.
