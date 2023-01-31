# %% Import Libraries
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy 
import pandas as pd
from sklearn import preprocessing
# %% Pre-Processing
df = pd.read_csv('data/sample.csv') 

le = preprocessing.LabelEncoder()
le.fit(df['intent'])
df['intent_label'] = le.transform(df['intent'])

data_texts = df["text"].to_list() # Features (not-tokenized yet)
data_labels = df["intent_label"].to_list() # Lables

train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))
# %% Compile and Train Model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=84)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy'])
model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16, validation_data=val_dataset.shuffle(1000).batch(16))

# %% Fine Tuning
# from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

# training_args = TFTrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )

# with training_args.strategy.scope():
#     trainer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# trainer = TFTrainer(
#     model=trainer_model,                 # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=val_dataset,             # evaluation dataset
# )

# %% Saving the model, the tokenizer and the classes
save_directory = "models/" 

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
numpy.save('models/classes.npy', le.classes_)

# %%
