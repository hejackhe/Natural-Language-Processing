#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

import numpy as np
from tensorflow import keras, argmax
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer

# Read training data
train_data = pd.read_csv('twitter-training-data.txt', sep='\\t', names=['tweet_id','sentiment','tweet_text'])
print("Training data has been read.")

print("Preprocessing data...")
# Prepare custom set of stopwords for removal
stopset = set(stopwords.words('english')) # remove stopwords
stopset.remove('not') # we keep not in our data to maintain negation
stopset.add('wo') # to account for won't -> "wo not" after regex
stopset.add('ca') # to account for can't -> "ca not" after regex
# Prepare stemmer
stemmer = PorterStemmer()

# Lowercase words
train_data['tweet_text'] = train_data['tweet_text'].str.lower()
# Remove URLs
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r'(((https?:\/\/)|(w{3}.))[\S]*)|([\w\d\/\.]*\.(com|cn|co|net|org|edu|uk|int|js|html))', '')
# Replace all n't or n' suffixes to "not" e.g "won't",'wouldn',"wouldn't" into "wo not", "would not"
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r'(n\'t|n\')', ' not ')
# Remove stopwords
train_data['tweet_text'].apply(lambda x: [item for item in x if item not in stopset])
# Replace all happy emojis to "happy" e.g :) :') ;) :D ;D :'D xD (: (': :') :3 c: C: c; C; c': C':
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r"\:\)|\:\'\)|\;\)|\:D|xD|\:3|\(\:|\('\:|\;D|\:\'D|c\:|C\:|c\;|C\;|c\'\:|C\'\:", ' happy ')
# Replace all sad emojis to "sad" e.g D; D: ): ); :'( D': Dx :( :'(' :c :C ;c ;C:'c:'C
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r"D\;|D\:|\)\:|\)\;|\:\'\(|D\'\:|Dx|\:\(|\:c|\:C|\;c|\;C|\:\'c|\:\'C", ' sad ')
# Remove twitter handles
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r'\@[\S]*', '')
# Remove numbers that are fully made of digits
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r'\b\d+\b','')
# Remove all non-alphanumeric characters except spaces.
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r'[^a-z0-9\s]','')
# Remove words with only 1 character. 
train_data['tweet_text'] = train_data['tweet_text'].str.replace(r'\b\w\b','')
# Stem all words
train_data['tweet_text'].apply(lambda x: [stemmer.stem(y) for y in x])

# sentiment becomes dependent variable
train_y = np.array(train_data['sentiment'])

print("Data preprocessing complete.")

for classifier in ['MaxEnt', 'NaiveBayes', 'SVC', 'LSTM']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'MaxEnt':
        print('Training ' + classifier)
        # extract features for training classifier1
        vectorizer = CountVectorizer(strip_accents='unicode', stop_words=stopset) # Count Vectorizer
        train_X = vectorizer.fit_transform(train_data.tweet_text)
        # train sentiment classifier1
        clf = linear_model.LogisticRegression(multi_class = 'ovr')
        clf.fit(train_X, train_y)
    elif classifier == 'NaiveBayes':
        print('Training ' + classifier)
        # extract features for training classifier2
        vectorizer = TfidfVectorizer(use_idf=True, strip_accents='unicode', stop_words=stopset) # TFIDF Vectorizer
        train_X = vectorizer.fit_transform(train_data.tweet_text)
        # train sentiment classifier2
        clf = naive_bayes.BernoulliNB()
        clf.fit(train_X, train_y)
    elif classifier == 'SVC':
        print('Training ' + classifier)
        # extract features for training classifier3
        vectorizer = TfidfVectorizer(use_idf=True, strip_accents='unicode', stop_words=stopset) # TFIDF Vectorizer
        train_X = vectorizer.fit_transform(train_data.tweet_text)
        # train sentiment classifier3
        clf = svm.LinearSVC(multi_class='crammer_singer')
        clf.fit(train_X, train_y)
    elif classifier == 'LSTM':
        print('Training ' + classifier)
        lb = LabelBinarizer(sparse_output=False)
        train_y = lb.fit_transform(train_data.sentiment)
        
        # Load pre-trained GloVe word-embedding vectors 
        embeddings_index = {}
        for i, line in enumerate(open('data/glove.6B.100d.txt')):
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
        
        # Initialise tokenizer 
        token = text.Tokenizer(num_words=5000)
        token.fit_on_texts(train_data['tweet_text'])
        word_index = token.word_index
        
        # Find longest tweet length
        train_lens = [len(x.split()) for x in train_data['tweet_text']]
        print("Max word count in training data: ", max(train_lens))
        
        # Convert text to padded sequence of tokens
        train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_data.tweet_text), maxlen=32) # Max word count of a tweet is 32 words
        
        # Create embedding matrix for embedding layer
        embedding_matrix = np.zeros((5000, 100))
        for word, i in word_index.items():
            if i < 5000:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        # Add an Input Layer
        input_layer = keras.layers.Input((32, ))
        # Add the word embedding Layer
        embedding_layer = keras.layers.Embedding(5000, 100, weights=[embedding_matrix], trainable=False)(input_layer)
        embedding_layer = keras.layers.SpatialDropout1D(0.3)(embedding_layer)
        # Add the LSTM Layer
        lstm_layer = keras.layers.LSTM(100)(embedding_layer)
        # Add the output Layers
        output_layer1 = keras.layers.Dense(32, activation="relu")(lstm_layer)
        output_layer1 = keras.layers.Dropout(0.25)(output_layer1)
        output_layer2 = keras.layers.Dense(3, activation="softmax")(output_layer1)
        # Compile the model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer2)
        model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['accuracy','categorical_accuracy'])
        # Fit the training dataset on the classifier
        model.fit(train_seq_x, train_y, epochs=15, batch_size=25)
    
    
    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        test_df = pd.read_csv(testset, sep='\\t', names=['tweet_id','sentiment','tweet_text'])

        # Test data is preprocessed just as training data was
        test_df['tweet_text'] = test_df['tweet_text'].str.lower()
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r'(((https?:\/\/)|(w{3}.))[\S]*)|([\w\d\/\.]*\.(com|cn|co|net|org|edu|uk|int|js|html))', '')
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r'(n\'t|n\')', ' not ')
        test_df['tweet_text'].apply(lambda x: [item for item in x if item not in stopset])
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r"\:\)|\:\'\)|\;\)|\:D|xD|\:3|\(\:|\('\:|\;D|\:\'D|c\:|C\:|c\;|C\;|c\'\:|C\'\:", ' happy ')
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r"D\;|D\:|\)\:|\)\;|\:\'\(|D\'\:|Dx|\:\(|\:c|\:C|\;c|\;C|\:\'c|\:\'C", ' sad ')
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r'\@[\S]*', '')
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r'\b\d+\b','')
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r'[^a-z0-9\s]','')
        test_df['tweet_text'] = test_df['tweet_text'].str.replace(r'\b\w\b','')
        test_df['tweet_text'].apply(lambda x: [stemmer.stem(y) for y in x])

        # Find longest tweet length
        #test_lens = [len(x.split()) for x in test_df['tweet_text']]
        #print("Max word count in test data: ", max(test_lens))

        predictions = {}
        if classifier != "LSTM":
            test_X = vectorizer.transform(test_df.tweet_text)
            preds = clf.predict(test_X)
            '''
            dev_y = np.array(test_df['sentiment'])
            acc_score = accuracy_score(dev_y, preds)
            conf_mat = confusion_matrix(dev_y, preds, labels = ["positive", "neutral", "negative"])
            f1 = f1_score(dev_y, preds, average='macro')
            a = classification_report(dev_y, preds)
            print(acc_score)
            print(conf_mat)
            print(f1)
            print(a)
            '''
            for i in range(len(preds)):
                predictions[str(test_df['tweet_id'][i])] = preds[i]
        else:
            # test_y = lb.fit_transform(test_df.sentiment)
            # test_y = test_y.argmax(axis=1)
            valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_df.tweet_text), maxlen=32)
            # predict the labels on validation dataset
            preds = model.predict(valid_seq_x)
            preds = preds.argmax(axis=1)
            for i in range(len(preds)):
                if preds[i] == 2:
                    predictions[str(test_df['tweet_id'][i])] = 'positive'
                elif preds[i] == 1:
                    predictions[str(test_df['tweet_id'][i])] = 'neutral'
                else:
                    predictions[str(test_df['tweet_id'][i])] = 'negative'
            # a = classification_report(test_y, preds)
            # print(a)
        #predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
