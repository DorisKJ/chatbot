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
from tensorflow.keras.layers import Dense, Input, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Reading the data file and axetracting the training data
f = open("intents.json")
file = json.loads(f.read())

text = []
label = []

for item in file['intents']:
    for text_id in item['patterns']:
        text.append(text_id)
        label.append(item['tag'])

#print(text)
#print(label)

# vectorizing
words = []
new_sentences = []

lemmatizer = WordNetLemmatizer()
for sentence in text:
    word = nltk.word_tokenize(sentence.lower())
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word])
    words.extend(word)
    new_sentences.append(lemmatized_output)
    
vectorizer = CountVectorizer()
vectorizer.fit(new_sentences)
#print(vectorizer.vocabulary_)
train_text = vectorizer.transform(new_sentences)
train_text = train_text.toarray()

train_label = vectorizer.transform(label)
train_label = train_label.toarray()
#print(vectorizer2.vocabulary_)
#words = sorted(list(set(words)))
print(len(train_text[0]))

# The model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_text[0]),), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(train_label[0]), activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#print(model.summary())
train_label = np.argmax(train_label, axis=1)
model.fit(train_text, train_label, epochs=100, batch_size=32, verbose=1)

# processing user input
def process_tool(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    words = [' '.join([lemmatizer.lemmatize(w) for w in tokens])]
    sentence_vect = vectorizer.transform(words)
    sentence_vect = sentence_vect.toarray()
    return sentence_vect

      #get the label for a given text
    
    index = labels[ind]
    return index

# Response generator
def response():
    while True:
        print("Me:", end="")
        msg = input()
        if msg.lower() == "bye":
            print("Chatty:", "See you next time.")
            break
            
        prediction = model.predict([process_tool(msg)])[0]
        ind = np.argmax(prediction)
        index = label[ind]
            
        for info in file["intents"]:
            if info['tag'] == index:
                print("Chatty:", np.random.choice(info['responses']))

print("You can chat with the bot now!")
print("(Enter 'bye to stop')")

response()
