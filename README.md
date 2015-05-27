# Description
This script generates descriptive statistics from Google Hangouts logs, 
including total words, words by speaker, etc. Hangouts transcripts should be
in plaintext format, extracted from JSON archives (accessible [here](https://www.google.com/settings/takeout)) with [hangouts-log-reader](https://bitbucket.org/dotcs/hangouts-log-reader/).

# Usage
<code>python chat-stats.py [TRANSCRIPT] [RESULTS_DIRECTORY]</code>

# Results
The script outputs a set of results files in the desired directory. 
summary.txt contains summary statistics for the file, including total words, time to 
read, words by speaker, and lexical diversity (types / tokens). words.txt lists every 
word used by each speaker, overall_ and speaker_word_frequencies.txt contain word 
frequencies sorted in descending order. overall_ and speaker_bigram_frequencies.txt 
contain bigram frequencies in descending order. speaker_timeseries contains the number
of words used per day by speaker. word_timeseries.txt contains occurrences of words of 
interest by day, by word. Finally, day_timeseries.txt contains number of words used per day.
