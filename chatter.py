import json
import string
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tflearn
import tensorflow.compat.v1 as tf
from keras.models import load_model
import random
import sys
from termcolor import colored


# open load requirements 
with open("dialog.json") as json_file:
    data = json.load(json_file)

with open("data.pickle", "rb") as data_file:
    words, tags, training_sentences, training_labels = pickle.load(data_file)

cv = pickle.load(open("vectorizer.pickel", "rb"))

model = load_model("chatty_model")

# Process input data

lemmatizer = WordNetLemmatizer()
#vectorizer = CountVectorizer()

#vecto = cv.transform(['How are you'])
#test2 = vecto.toarray()

#print(test2.shape)

def process_tool(sentence):
    tokens = nltk.word_tokenize(sentence)
    words = [' '.join([lemmatizer.lemmatize(w) for w in tokens])]
    sentence_vect = cv.transform(words)
    sentence_vect = sentence_vect.toarray()
    #new_stce = np.argmax(new_stce, axis=1)
    return sentence_vect


# Prediction and responses

def response():
    # Configuring the chatbot responses
    while True:
        print(colored("Me: ", 'blue') , end="")
        msg = input()
        if msg.lower() == "bye":
            print(colored("Chatty:", "green"), "Good talking to you. See you next time.")
            break

        test = model.predict([process_tool(msg)])[0]

        ind = np.argmax(test)
        index = tags[ind]

        for info in data["objects"]:
            if info['tag'] == index:
                print(colored("Chatty:", 'green'), np.random.choice(info['responses']))


print(colored("You can chat with the bot now!", 'red', attrs=['bold']))
print("(Enter 'bye to stop')")

response()


















