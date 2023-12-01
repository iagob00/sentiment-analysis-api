from preprocessing import CustomPreprocess
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

movie_reviews = pd.read_csv('a1_IMDB_Dataset.csv')

custom = CustomPreprocess()
processed_reviews = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    processed_reviews.append(custom.preprocess_text(sen))

sentiment = movie_reviews['sentiment']
sentiment = np.array(list(map(lambda x: 1 if x=="positive" else 0, sentiment)))

reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(processed_reviews, sentiment, test_size=0.20, random_state=42)

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(reviews_train)

reviews_train = word_tokenizer.texts_to_sequences(reviews_train)
reviews_test = word_tokenizer.texts_to_sequences(reviews_test)
import io
import json
tokenizer_json = word_tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

vocab_length = len(word_tokenizer.word_index) + 1
maxlen = 100

reviews_train = pad_sequences(reviews_train, padding='post', maxlen=maxlen)
reviews_test = pad_sequences(reviews_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
embedding_matrix = np.zeros((vocab_length, 100))
for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

lstm_model = Sequential()
embedding_layer = Embedding(vocab_length, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

lstm_model.add(embedding_layer)
lstm_model.add(LSTM(128))

lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(lstm_model.summary())
lstm_model_history = lstm_model.fit(reviews_train, sentiment_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = lstm_model.evaluate(reviews_train, sentiment_train, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(lstm_model_history.history['acc'])
plt.plot(lstm_model_history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(lstm_model_history.history['loss'])
plt.plot(lstm_model_history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

lstm_model.save(f"./lstm_model_acc_{round(score[1], 3)}.h5", save_format='h5')
