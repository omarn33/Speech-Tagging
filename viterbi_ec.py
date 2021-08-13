# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math
from collections import Counter

def viterbi_ec(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    prediction = model(test, train)
    return prediction


'''
Finds all the tags and words within a training set
'''


def find_unique_pairs(dataset):
    # Declare a set to store the tags/words within the dataset (without duplicates)
    unique_tags = set()
    unique_words = set()

    for sentence in dataset:
        for pair in sentence:
            unique_words.add(pair[0])
            unique_tags.add(pair[1])

    # Remove START and END keywords
    unique_words.remove('START')
    unique_words.remove('END')
    unique_tags.remove('START')
    unique_tags.remove('END')

    return unique_words, unique_tags


'''
Calculates the emission table
'''


def calculate_emission_table(dataset, tags):
    emission_table = {}

    # For each tag, count the number of each word
    for tag in tags:
        counter = Counter()
        words = []

        for sentence in dataset:
            for pair in sentence:
                if pair[1] == tag:
                    words.append(pair[0])

        counter.update(words)
        emission_table[tag] = counter

    return emission_table


'''
Calculates the tag with highest frequency
'''


def calculate_most_freq_tag(emission_table):
    tag_count = {}

    for part in emission_table:
        tag_count[part] = sum(emission_table[part].values())

    # Return part of speech that had highest count
    return max(tag_count, key=tag_count.get)


'''
Calculates the tag with highest probability given a word
'''


def calculate_most_likely_tag(emission_table, word):
    tag_count = {}

    for part in emission_table:
        tag_count[part] = emission_table[part][word]

    # Return part of speech that had highest count
    return max(tag_count, key=tag_count.get)


def model(test, train):
    unique_words, unique_tags = find_unique_pairs(train)
    emission_table = calculate_emission_table(train, unique_tags)
    most_freq_tag = calculate_most_freq_tag(emission_table)

    prediction = []

    for sentence in test:
        sentence_prediction = [("START", "START")]
        for word in sentence:
            if word != "START" and word != "END":
                if word in unique_words:
                    tag = calculate_most_likely_tag(emission_table, word)
                    # print(word, tag)
                else:
                    tag = most_freq_tag
                    # print(word, tag)

                sentence_prediction.append((word, tag))

        sentence_prediction.append(("END", "END"))
        prediction.append(sentence_prediction)

    return prediction