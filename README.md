# Description
This script generates descriptive statistics from Google Hangouts logs, 
including total words, words by speaker, etc. Hangouts transcripts should be
in plaintext format, extracted from JSON archives with [hangouts-log-reader](https://bitbucket.org/dotcs/hangouts-log-reader/).

# Usage
<code>python chat-stats.py [TRANSCRIPT] [RESULTS_DIRECTORY]</code>

# Results
The script outputs a set of results files in the desired directory. 
summary.txt contains summary statistics for the file (total words, time to 
read, words by speaker, etc.). words.txt lists every word used by each speaker,
overall_ and speaker_word_frequencies.txt contain word frequencies sorted in
descending order. overall_ and speaker_bigram_frequencies.txt contain bigram
frequencies in descending order. word_timeseries.txt contains occurrences of
words of interest by day, by word. Finally, day_timeseries.txt contains number
of words used per day.
