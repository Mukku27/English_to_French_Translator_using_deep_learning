
"""
model for language translater
Programmed by Mukesh Vemulapalli 

* 2024-04-03: Initial coding
* 2022-04-07: Small revision of code, checked that it works with latest TensorFlow version
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Data loading and preprocessing functions
data_path = "/content/fra.txt"

def to_lines(text):
    # Splitting the text into sentences
    sents = text.strip().split('\n')
    # Splitting each sentence into source and target language pairs
    sent = [i.split('\t') for i in sents]
    return sent

# Reading data from file
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Converting text data into language pairs
fra_eng = to_lines(data)

# Converting the list of language pairs into a numpy array
fra_eng = np.array(fra_eng)

# Preprocessing function to remove punctuation and convert to lowercase
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Applying preprocessing to source and target language pairs
fra_eng[:, 0] = [preprocess_text(s) for s in fra_eng[:, 0]]
fra_eng[:, 1] = [preprocess_text(s) for s in fra_eng[:, 1]]

# Tokenizing source and target language sentences
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(fra_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = 8

fra_tokenizer = Tokenizer()
fra_tokenizer.fit_on_texts(fra_eng[:, 1])
fra_vocab_size = len(fra_tokenizer.word_index) + 1
fra_length = 8

# Encoding sequences
def encode_sequence(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

# Splitting data into train and test sets
train, test = train_test_split(fra_eng, test_size=0.2, random_state=12)

trainX = encode_sequence(fra_tokenizer, fra_length, train[:, 1])
trainY = encode_sequence(eng_tokenizer, eng_length, train[:, 0])

testX = encode_sequence(fra_tokenizer, fra_length, test[:, 1])
testY = encode_sequence(eng_tokenizer, eng_length, test[:, 0])

# Defining the sequence-to-sequence model
def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

# Compiling the model
model = define_model(fra_vocab_size, eng_vocab_size, fra_length, eng_length, 512)
rms = optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

# Model training
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1]), epochs=200, batch_size=512, validation_split=0.25)

# Making predictions
pred = model.predict_classes(testX.reshape(testX.shape[0], testX.shape[1]))

# Decoding function to convert predicted integers into corresponding words
def decode_sequence(model, tokenizer, data):
    sequences = pad_sequences(data, maxlen=data.shape[1], padding='post')
    preds = model.predict_classes(sequences)
    decoded = []
    for i in range(len(preds)):
        sentence = []
        for j in range(len(preds[i])):
            word = get_word(preds[i][j], tokenizer)
            if word is None:
                continue  # Skip if word not found
            sentence.append(word)
        decoded.append(" ".join(sentence))
    return decoded

# Decoding the predicted sequences
decoded_sentences = decode_sequence(model, eng_tokenizer, testX)

# Printing predicted and actual sentences
for i in range(len(decoded_sentences)):
    print('English:', decoded_sentences[i])
    print('Actual:', testY[i])
    print()

# Saving the trained model
model.save('seq2seq_model.h5')

