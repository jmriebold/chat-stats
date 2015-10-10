# -*- coding: utf-8 -*-

""" 
This script generates descriptive statistics from Google Hangouts logs, 
including total words, words by speaker, etc. Hangouts transcripts should be
in plaintext format, extracted from JSON archives (accessible at 
https://www.google.com/settings/takeoutwith hangouts-log-reader 
(https://bitbucket.org/dotcs/hangouts-log-reader/).

Usage:
    python chat-stats.py [TRANSCRIPT] [RESULTS_DIRECTORY]

The script outputs a set of results files in the desired directory. 
summary.txt contains summary statistics for the file, including total words,
time to read, words by speaker, and lexical diversity (types / tokens). 
words.txt lists every word used by each speaker, overall_ and 
speaker_word_frequencies.txt contain word frequencies sorted in descending 
order. overall_ and speaker_bigram_frequencies.txt contain bigram frequencies
in descending order. speaker_timeseries contains the number of words used per
day by speaker. word_timeseries.txt contains occurrences of words of interest
by day, by word. Finally, day_timeseries.txt contains number of words used 
per day.
"""

from __future__ import division
import sys
import os
import re
from collections import defaultdict, Counter
import datetime
import random

import numpy


# Extract and return name of message sender from line
def process_name(line):
    name = re.sub("^.*\d{2}:\d{2}:\d{2}: ", "", line)
    name = re.sub("^<(\w+\s*\w*\s*\w*)>.*", "\\1", name).strip()

    return name


# Return set of chat participants
def get_participants(transcript):
    with open(transcript, 'r', encoding="utf8") as in_file:
        names = list(set(process_name(line).lower() for line in
                         in_file.readlines() if re.match("^\[hangouts.py\]", line) is not
                         None))

    return names


# Initialize matrix for timeseries data
def initialize_table(nrows, row_names, ndays):
    # Initialize table as 0s
    table = numpy.zeros((nrows, ndays + 2), dtype=object)

    # Set row names
    for row, array in enumerate(table):
        table[row, 0] = row_names[row]

    return table


# Find and return timestamp of message
def get_datetime(line):
    timestamp = re.sub(".*(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*", "\\1",
                       line).strip()
    timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    return timestamp


# Calculate and return timedelta between two dates
def calculate_days(start_date, end_date):
    date_delta = datetime.datetime.date(end_date) - start_date

    return date_delta.days


# Calculate and return the time of day in 6ths of an hour for a timestamp
def calculate_daytime(timestamp):
    hours = datetime.datetime.strftime(timestamp, "%H")
    minutes = datetime.datetime.strftime(timestamp, "%M")

    return (int(hours) * 60 + int(minutes)) // 10


# Remove hyperlinks, separate words with hyphens, slashes, ellipses, split on
# whitespace
def process_message(message):
    message = re.sub("http[^\n\s]*", "", message)

    return re.sub("-+|/+|\.{2}", " ", message).split()


# Strip punctuation and whitespace, retain contracted forms, lemmatize highly
# variable words
def process_word(word):
    # Contracted forms to spare from strip()ing
    contracted_forms = [
        "can't", "could've", "couldn't", "didn't", "doesn't", "don't",
        "hadn't", "hasn't", "haven't", "he'd", "he'll", "here's", "he's",
        "i'd", "i'll", "i'm", "i've", "isn't", "it'd", "it'll", "it's",
        "let's", "she'd", "she'll", "she's", "that'd", "that'll", "that's",
        "there's", "there'll", "they're", "this'd", "this'll", "wasn't",
        "we'd", "we're", "we've", "what'd", "what'll", "what's", "won't",
        "would've", "wouldn't", "you'd", "you'll", "you're", "you've"
    ]

    word = word.strip()

    # Remove punctuation
    if word not in contracted_forms:
        word = re.sub("\W*", "", word)

    return word


# Return list of bigrams in message
def getbigrams(message):
    return [word + ' ' + message[i + 1] for i, word in enumerate(message) if i + 1 <= len(message) - 1]


# Return list of trigrams in message, padded with BOS/EOS
def gettrigrams(message):
    message.append('EOS')
    message.append('EOS')
    message.insert(0, 'BOS')
    message.insert(0, 'BOS')

    return [word + ' ' + message[i + 1] + ' ' + message[i + 2] for i, word in enumerate(message) if
            i + 2 <= len(message) - 1]


# Use trigram Markov chains to generate random text from a participant
def generatetext(name):
    trigrams = speaker_trigrams[name]

    # Get sentence start
    bos = [trigram for trigram in trigrams if trigram.split()[0] == 'BOS']

    # Select trigram
    next = bos[random.randint(0, len(bos) - 1)]

    # Add to sentence
    sentence = next.split()[0] + ' '

    # Build sentence
    while next.split()[2] != 'EOS':
        # Get possible trigram transitions
        choices = [trigram for trigram in trigrams if trigram.split()[0:2] == next.split()[1:3]]

        # Get next trigram
        next = choices[random.randint(0, len(choices) - 1)]

        # Add to sentence
        sentence = sentence + next.split()[0] + ' '

    # Add remaining words
    sentence = sentence + next.split()[1] + ' ' + next.split()[2]

    # Remove BOS/EOS, strip whitespace
    sentence = re.sub('BOS|EOS', '', sentence).strip()

    # Add period
    sentence += '.'

    sentence = name.title() + ': ' + sentence.capitalize()

    # Capitalize "I"
    sentence = re.sub('i(?=\')', 'I', sentence)
    sentence = re.sub(' i ', ' I ', sentence)

    return sentence


# Words of interest, edit this list to get timeseries data for these words
# (e.g. for plotting over time, etc.)
keywords = ["a", "b", "c"]

# Stop list to exclude function words, etc.
stop_list = [
    "a", "about", "after", "again", "all", "also", "am", "an", "and",
    "another", "any", "are", "as", "at", "be", "because", "behind", "been",
    "being", "but", "by", "came", "can", "can't", "come", "could",
    "couldn't", "could've", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "else", "even", "few", "for", "from", "get", "getting", "gets",
    "go", "goes", "going", "gonna", "good", "got", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "hers", "him", "his", "how", "i", "i'd", "i'll", "if",
    "i'm", "in", "inside", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
    "i've", "just", "know", "let's", "like", "me", "my", "naw", "no", "not",
    "now", "oh", "of", "off", "ok", "okay", "on", "one", "or", "our", "out",
    "outside", "really", "right", "she", "she'd", "she'll", "she's", "should",
    "that", "that'd", "that'll", "that's", "the", "their", "them", "then",
    "there", "they", "these", "those", "think", "this", "this'd", "this'll",
    "they're", "so", "some", "though", "to", "up", "us", "very", "was", "we",
    "well", "went", "we're", "were", "we've", "what", "what's", "what'd",
    "what'll", "when", "which", "while", "who", "why", "will", "with",
    "without", "would", "wouldn't", "would've", "yeah", "yes", "you", "you'd",
    "you'll", "your", "you're", "you've"
]

# Counters for words/bigrams/etc.
speaker_words = defaultdict(Counter)
speaker_totals = Counter()
speaker_bigrams = defaultdict(Counter)
speaker_bigram_totals = Counter()
speaker_trigrams = defaultdict(Counter)
overall_words = Counter()
overall_bigrams = Counter()

# Store transcript file
try:
    transcript = sys.argv[1]
except IndexError:
    raise IOError("Transcript file does not exist.")

# Store results dir
try:
    results_dir = sys.argv[2]
except IndexError:
    raise IOError("Please enter a results directory.")

# Ensure dir ends with /
if results_dir[-1] != '/':
    results_dir += '/'

# Create directory if necessary
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Get list of participants in chat
names = get_participants(transcript)

# Calculate daterange
with open(transcript, 'r', encoding="utf8") as in_file:
    lines = list(in_file)
    first = lines[0]
    n = -1
    last = lines[n]

    while last[0] != '[':
        n -= 1
        last = lines[n]

    start = list(map(int, first.split(' ')[1].split('-')))
    end = list(map(int, last.split(' ')[1].split('-')))

start_date = datetime.date(start[0], start[1], start[2])
end_date = datetime.date(end[0], end[1], end[2])
date_delta = end_date - start_date

ndays = date_delta.days

# Initialize timeseries tables
speaker_timeseries = initialize_table(len(names), names, ndays)
word_timeseries = initialize_table(len(keywords), keywords, ndays)
day_timeseries = numpy.zeros((24 * 6, 1), dtype=object)

previous_timestamp = ""

with open(transcript, 'r', encoding="utf8") as in_file:
    for line in in_file.readlines():
        line = line.lower()

        # Use prepended script name to differentiate between single- and
        # multi-line messages
        if re.match("^\[hangouts.py\]", line) is not None:
            # Get timestamp of message
            timestamp = get_datetime(line)

            # Update previous timestamp
            if previous_timestamp == "":
                previous_timestamp = timestamp

            # Get speaker name
            name = process_name(line)

            # Get list of words in message and remove formatting
            message = process_message(re.sub(".*<.*> (.*)", "\\1", line))

        else:
            message = process_message(line)

        # Process words in message and remove blanks
        message = [process_word(word) for word in message if process_word(word) is not None]

        # Add words to Counters
        for word in message:
            speaker_words[name][word] += 1

            # Update timeseries data for keywords
            if word in keywords:
                word_timeseries[keywords.index(word), calculate_days(start_date, timestamp) + 1] += 1

        # Update timeseries data for speakers
        speaker_timeseries[names.index(name), calculate_days(start_date, timestamp) + 1] \
            += len(message)

        # Update day timeseries
        day_timeseries[calculate_daytime(timestamp), 0] += len(message)

        # Generate list of bigrams in message
        bigrams = getbigrams(message)

        # Add bigrams to Counter
        for bigram in bigrams:
            speaker_bigrams[name][bigram] += 1

        # Generate list of trigrams in message
        trigrams = gettrigrams(message)

        # Add trigrams to Counter
        for trigram in trigrams:
            speaker_trigrams[name][trigram] += 1

        previous_timestamp = timestamp

# Get speaker totals
for speaker in names:
    speaker_totals[speaker] = sum(speaker_words[speaker].values())

# Get overall word frequencies
for speaker in names:
    overall_words += speaker_words[speaker]

# Get overall word frequencies
for speaker in names:
    overall_bigrams += speaker_bigrams[speaker]

# Get total number of words
overall_total = sum(overall_words.values())

# Print summary stats
with open(results_dir + "summary.txt", 'w', encoding="utf8") as out_file:
    out_file.write("GENERAL\n-------\n")
    out_file.write("total length: " + str(overall_total) + " words" + "\ntime to read: " + str(
        round(overall_total / 250 / 60, 2)) + " hours\n\n")

    out_file.write("WORDS\n-----\n")
    for speaker, total in speaker_totals.most_common():
        out_file.write(speaker + "\t" + str(total) + "\t" + str(round(float(
            total / overall_total * 100), 2)) + "%\n")
    out_file.write("\n")

    out_file.write("LEXICAL DIVERSITY\n-----------------\n")
    for speaker in names:
        out_file.write(speaker + "\t" + str(round(float(len(
            speaker_words[speaker].keys()) / speaker_totals[speaker]), 2)) + "\n")

# Save timeseries data
numpy.savetxt(results_dir + "speaker_timeseries.txt", speaker_timeseries,
              delimiter="\t", fmt="%s")
numpy.savetxt(results_dir + "word_timeseries.txt", word_timeseries,
              delimiter="\t", fmt="%s")
numpy.savetxt(results_dir + "day_timeseries.txt", day_timeseries,
              delimiter="\t", fmt="%s")

# Save words
with open(results_dir + "words.txt", 'w', encoding="utf8") as out_file:
    for speaker in names:
        for word, count in speaker_words[speaker].most_common():
            out_file.write("%s\t%s\n" % (speaker, word))

# Save overall word frequencies
with open(results_dir + "overall_word_frequencies.txt", 'w', encoding="utf8") as out_file:
    for word, count in overall_words.most_common():
        if count > 1 and word not in stop_list and re.match("^\d+$", word) \
                is None:
            out_file.write("%s\t%s\t%s\n" % (word, count, round(count /
                                                                speaker_totals[speaker] * 100, 3)))

# Save speaker word frequencies
with open(results_dir + "speaker_word_frequencies.txt", 'w', encoding="utf8") as out_file:
    for speaker in names:
        for word, count in speaker_words[speaker].most_common():
            if count > 1 and word not in stop_list and re.match("^\d+$", word) \
                    is None:
                out_file.write("%s\t%s\t%s\t%s\n" % (speaker, word, count,
                                                     round(count / speaker_totals[speaker] * 100, 3)))

# Save overall bigram frequencies
with open(results_dir + "overall_bigram_frequencies.txt", 'w', encoding="utf8") as out_file:
    for bigram, count in overall_bigrams.most_common():
        if count > 1:
            out_file.write("%s\t%s\t%s\n" % (bigram, count, round(count /
                                                                  speaker_totals[speaker] * 100, 3)))

# Save speaker bigram frequencies
with open(results_dir + "speaker_bigram_frequencies.txt", 'w', encoding="utf8") as out_file:
    for speaker in names:
        for bigram, count in speaker_bigrams[speaker].most_common():
            if count > 1:
                out_file.write("%s\t%s\t%s\t%s\n" % (speaker, bigram, count,
                                                     round(count / speaker_totals[speaker] * 100, 3)))
