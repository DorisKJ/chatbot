import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
#import random
import numpy as np
import pickle
import tflearn
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout#, Activation 
from tensorflow.keras.models import Sequential


# Reading the data file and axetracting the training data
file = open("dialog.json")
intents = json.loads(file.read())

sentences = []
tags = []
responses = []

for intent in intents['objects']:
    for sentence in intent['patterns']:
        sentences.append(sentence) # training data
        tags.append(intent['tag'])
    #responses.append(intent['responses'])

    if intent['tag'] not in tags:
            tags.append(intent['tag'])


words = []
new_sentences = []

#Processing the training data
#No  lemma without token
lemmatizer = WordNetLemmatizer()
for sentence in sentences:
    #sentence = "".join([wrd for wrd in sentence if wrd not in string.punctuation])
    # Tokenizing: breaking sentences to create a list of words
    word = nltk.word_tokenize(sentence.lower())
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word])
    words.extend(word)
    new_sentences.append(lemmatized_output)
    # remove stopwords (not really necessary here but could be usefull for larger data)
    #words = [wrd for wrd in words if wrd not in stopwords.words('english')]
#print(sentences)
#print(words)
# Vectorizing everything
#CountVectorizer by default will tokenizer and convert text to lowercase. stopword is used by default if sentence in english

cv = CountVectorizer()
#training_sentences = cv.fit_transform(new_sentences)
training_sentences = cv.fit(new_sentences)

pickle.dump(cv, open("vectorizer.pickel", "wb"))

training_sentences = cv.transform(new_sentences)
training_sentences = training_sentences.toarray()

#print(training_sentences.shape)
#print(test1.shape)

training_labels = cv.transform(tags)
training_labels = training_labels.toarray()

#print(len(training_labels))

words = sorted(list(set(words)))

#print(words)

# Write everything to disc
with open("data.pickle","wb") as f:
    pickle.dump((words, tags, training_sentences, training_labels),f)


# Training model

model = Sequential()
model.add(Dense(128, input_shape=(len(training_sentences[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(training_labels[0]), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

#model.summary()


training_labels = np.argmax(training_labels, axis=1)
history = model.fit(training_sentences, training_labels, epochs=500, batch_size=5, verbose=1)
model.save("chatty_model")








