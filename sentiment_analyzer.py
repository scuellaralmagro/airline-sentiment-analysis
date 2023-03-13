# Sentiment Analyzer Script
# Description: This script will analyze the sentiment of a given tweet. It is trained on a dataset of 14640 tweets about
# airlines. The script will take a tweet as input and output its sentiment as positive, negative, or neutral.
# Author: @scuellaralmagro
# Date: March 2023

import pandas as pd
import joblib
import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

## Download the necessary NLTK resources

nltk.download('stopwords')
nltk.download('wordnet')

## Helper Functions

def tokenize(tweet):
    """Tokenize a tweet into words"""
    return word_tokenize(tweet)

def remove_stopwords(tweet):
    """Remove stopwords from a tweet"""
    return [word for word in tweet if word not in stopwords.words('english')]

def remove_punctuation(tweet):
    """Remove punctuation from a tweet"""
    return [word for word in tweet if word.isalpha()]

def input_tweet():
    """Get a tweet from the user"""
    return input('Enter a tweet: ')

def lemmatize(tweet):
    """Lemmatize a tweet"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tweet]

def extract_features(tweet):
    """Extract features from a tweet"""
    words = set(tweet.split())
    features = {word : (word in words) for word in words}

    return features

## Main Function

def main():
    # Load the model
    print('-- Welcome to the Airline Sentiment Analyzer --')
    print('-- Loading model... --')
    model = joblib.load('svm_classifier.joblib')
    print('-- Model loaded --')

    # Get a tweet from the user
    tweet = input_tweet()

    # Start the timer
    start = time.time()

    # Preprocess the tweet
    print('-- Preprocessing tweet... --')
    tweet = tokenize(tweet)
    tweet = remove_stopwords(tweet)
    tweet = remove_punctuation(tweet)
    tweet = lemmatize(tweet)
    tweet = ' '.join(tweet)
    print('-- Tweet preprocessed --')

    # Vectorize the tweet
    print('-- Vectorizing tweet... --')
    tweet = extract_features(tweet)
    print('-- Tweet vectorized --')

    # Predict the sentiment of the tweet
    print('-- Predicting sentiment... --')
    sentiment = model.classify(tweet)
    print('-- Sentiment predicted --')

    # Print the sentiment of the tweet
    print(f'\n-- The sentiment of the tweet is {sentiment.upper()} --\n')

    # Stop the timer
    end = time.time()

    # Print the time it took to analyze the tweet
    print(f'-- Analysis complete in {round((end - start), 2)} seconds --')

## Run the main function if the script is run directly and not imported.

if __name__ == '__main__':
    main()