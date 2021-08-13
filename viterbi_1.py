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
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
from collections import Counter

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    prediction = model(test, train, laplace=0.0001)
    return prediction


def find_unique_pairs(dataset):
    """
    Finds all words and tags in the training dataset (without duplicates)

    @param list dataset: the training data set
    @return list unique_words: all words in the dataset
            list unique_tags: all tags in the dataset
    """

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

    return list(unique_words), list(unique_tags)


def find_unique_test_words(dataset):
    """
    Finds all the words in the test dataset (without duplicates)

    @param list dataset: the test dataset
    @return list unique_test_words: all words in the test dataset
    """

    unique_test_words = set()

    for sentence in dataset:
        for word in sentence:
            unique_test_words.add(word)

    unique_test_words.remove('START')
    unique_test_words.remove('END')

    return list(unique_test_words)


def calculate_emission_probability(dataset, unique_words, unique_test_words, tags, laplace):
    """
    Calculates the emission probability table for the Viterbi Algorithm

    @param list dataset: the training set
    @param list unique_words: all the words in the training set (w/o duplicates)
    @param list unique_test_words: all the words in the test set (w/o duplicates)
    @param list tags: all the tags in the training set
    @param double laplace: the Laplace smoothing parameter
    @return Dict of Counters (dicts) that stores the emission probability
    """

    # Dictionary of Counters to store the probability for each word in each tag
    emission_probability = {}

    # Remove START and END keyword
    tags.remove('START')
    tags.remove('END')

    # Calculate unseen words: words that are present in the test dataset but not in the training set
    unseen_words = list(set(unique_test_words) - set(unique_words))

    # For each tag, count the frequency of each word in the training set
    for tag in tags:
        counter = Counter()
        words = []

        # Store the words in each sentence for a given tag
        for sentence in dataset:
            for pair in sentence:
                if pair[1] == tag:
                    words.append(pair[0])

        # Count the frequency of each word
        counter.update(words)

        # Add words from the training set with a frequency of zero
        counter.update(list(set(unique_words) - set(words)))
        counter.subtract(list(set(unique_words) - set(words)))

        # Add unseen words from the test dataset with a frequency of zero
        counter.update(unseen_words)
        counter.subtract(unseen_words)

        # Store Counter (dict) for each tag
        emission_probability[tag] = counter

    # Display Word Frequency Per Tag
    # print("Word Frequency:")
    # print(emission_probability)
    # print()

    # Calculate number of words for each tag
    number_of_words = len(emission_probability[tags[0]].values())

    # Calculate Probabilities with Laplace Smoothing
    for tag in tags:
        tag_sum = sum(emission_probability[tag].values())

        for word in emission_probability[tag]:
            emission_probability[tag][word] = math.log10(
                (emission_probability[tag][word] + laplace) / (tag_sum + (number_of_words * laplace)))

    # Display Probability
    # print("Emission Probability:")
    # print(emission_probability)
    # print()

    return emission_probability


def calculate_transition_probability(dataset, tags, laplace):
    """
    Calculates the transition probability table for the Viterbi Algorithm

    @param list dataset: the training set
    @param list tags: all the tags in the training set
    @param double laplace: the Laplace smoothing parameter
    @return Dict of Counters (dicts) that stores the transition probability
    """

    # A list that stores all the tags in the order of the training set with words
    # ex: ["START", "NOUN", "MODAL", ... ,"END", "START", "ADJ"...]
    flatten_tags = []

    for sentence in dataset:
        for pair in sentence:
            flatten_tags.append(pair[1])

    # Display Flatten Tags
    # print("Flatten Tags:")
    # print(flatten_tags)
    # print()

    # Dictionary of Counters to store the probability of the following tags for each tag
    transition_probability = {}

    # For each tag, find the following tag
    for tag in tags:
        counter = Counter()
        tag_pairs = []

        for tag_index in range(len(flatten_tags) - 1):
            if tag == flatten_tags[tag_index]:
                tag_pairs.append(flatten_tags[tag_index + 1])

        counter.update(tag_pairs)

        # Add tags from the training set with a frequency of zero
        freq_zero_tags = list((set(flatten_tags) - set(tag_pairs)))
        counter.update(freq_zero_tags)
        counter.subtract(freq_zero_tags)

        # Store Counter (dict) for each tag
        transition_probability[tag] = counter

    # Display Tag Frequency Per Tag
    # print("Tag Frequency:")
    # print(transition_probability)
    # print()

    # Drop ("START, "START") and ("START", "END") to prevent probability calculation error
    for tag in tags:
        for tag_pair in transition_probability[tag].copy():
            if (tag_pair == "START") or (tag == "START" and tag_pair == "END"):
                del transition_probability[tag][tag_pair]

    # Display Tag Frequency Per Tag
    # print("(Corrected) Tag Frequency:")
    # print(transition_probability)
    # print()

    # Calculate number of tags for each tag
    number_of_tags = len(transition_probability[tags[0]].values())

    # Calculate Probabilities with Laplace Smoothing
    for tag in tags:
        tag_sum = sum(transition_probability[tag].values())

        for tag_pair in transition_probability[tag]:
            transition_probability[tag][tag_pair] = math.log10(
                (transition_probability[tag][tag_pair] + laplace) / (tag_sum + (number_of_tags * laplace)))

    # Drop END
    del transition_probability['END']

    # Display Probability
    # print("Transition Probability:")
    # print(transition_probability)
    # print()

    return transition_probability


def model(test, train, laplace):
    # Find all words and tags in training dataset
    unique_words, unique_tags = find_unique_pairs(train)

    # Find all words in test dataset
    unique_test_words = find_unique_test_words(test)

    # Calculate the emission probability
    emission_probability = calculate_emission_probability(train, unique_words, unique_test_words, unique_tags.copy(),
                                                          laplace)
    # Calculate the transition probability
    transition_probability = calculate_transition_probability(train, unique_tags, laplace)

    print(transition_probability["START"]["DET"])

    prediction = []

    # for sentence in test:
    #   sentence_prediction = [("START", "START")]
    #   for word in sentence:
    #     if word != "START" and word != "END":
    #       if word in unique_words:
    #         tag = calculate_most_likely_tag(emission_table, word)
    #         #print(word, tag)
    #       else:
    #         tag = most_freq_tag
    #         #print(word, tag)

    #       sentence_prediction.append((word, tag))

    #   sentence_prediction.append(("END", "END"))
    #   prediction.append(sentence_prediction)

    return prediction



