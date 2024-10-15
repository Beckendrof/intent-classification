# DistilBERT Transformer model for Multi-class Classification
For this project I have finetuned a DistilBERT transofrmer model to perform intent classification.

## Prerequisities

The dataset used for finetuning comprises of 84 classes of general conversational responses. This dataset however can be discarded and a new dataset can be used in place. Just make sure to change the `file_name` variable in the `train.py` file to the name of your csv file. The dataset is in csv format and should contain two columns titled `text` and `intent`. 

Before running the `train.py` file make sure to clear all the files from the `models` folder.

## Installation

Run the requirements.txt file first to install all dependencies (preferably in a virtual environment).

    pip install -r requirements.txt

Run the `main.py` file directly or run the `train.py` to re-train the model with new data. Since the DistilBERT version is fast and easy to deploy, the training time period is fairly low. After training is over run the `main.py` to execute the program.


