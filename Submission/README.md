This project implements a sequential recommendation model based on BERT4Rec to predict next-item interactions from user behavior sequences. Below is a detailed introduction to the project modules.

### File Descriptions:

- **`train_ratio.py`**: 
  - This script handles the model training process. During training, it outputs evaluation results on the validation set and evaluates the model’s performance on the test set after completion.

- **`model1.py`**: 
  - Contains the implementation of a model using HuggingFace’s BERT architecture.

- **`model2.py`**: 
  - Implements the model architecture created from scratch.

- **`maskdataset.py`**: 
  - This script applies random masking to the training dataset.

- **`evaluation_ratio.py`**: 
  - Contains functions to compute evaluation metrics NDCG@10 and Recall@10 using a rolling window approach.

- **`data_load_ratio.py`**: 
  - This script loads and processes the dataset, then splits it into three parts: 70% for training, 15% for validation, and 15% for testing.


## Run the Code
Run train_ratio.py and it will start the model training, output the changes of the validation set during the training process, and finally output the evaluation on the test set.
