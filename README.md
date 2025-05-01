This project implements a sequential recommendation model based on BERT4Rec to predict next-item interactions from user behavior sequences. Below is a detailed introduction to the project modules.

### File Descriptions:

- **`train_ratio.py`**: 
  - This script handles the model training process. During training, it outputs metric changes on the validation set and evaluates the model’s performance on the test set after completion.

- **`model1.py`**: 
  - Contains the implementation of a model using HuggingFace’s BERT architecture for training and fine-tuning on a provided dataset.

- **`model2.py`**: 
  - Implements a custom model architecture created from scratch, designed for comparison with the BERT model.

- **`maskdata.py`**: 
  - This script applies random masking to the training dataset, which can be useful for certain types of data augmentation or regularization.

- **`evaluation_ratio.py`**: 
  - Contains functions to compute evaluation metrics such as NDCG@10 and Recall@10 using a rolling window approach, which is often used in recommendation systems.

- **`data_load_ratio.py`**: 
  - This script loads and processes the dataset, then splits it into three parts: 70% for training, 15% for validation, and 15% for testing.

- **`README.md`**: 
  - The project description file, providing an overview of the project and instructions on how to use the various scripts in the repository.

## Run the Code
Run train_ratio.py and it will start the model training, output the indicator changes of the validation set during the training process, and finally output the indicator evaluation on the test set.
