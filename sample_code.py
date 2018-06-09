import nltk
from urllib import request
from nltk.sentiment.vader import SentimentIntensityAnalyzer


'''
Resources:
NLTK: https://www.nltk.org/book/
NLTK: https://textminingonline.com/dive-into-nltk-part-i-getting-started-with-nltk
Sentiment analysis: https://opensourceforu.com/2016/12/analysing-sentiments-nltk/
'''

# How to store the contents of a .txt URL into a string and then tokenize it for analysis
url = "https://www.gutenberg.org/files/768/768.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)

# How to do a concordance of a word (e.g. "moors")
res = text.concordance("moors")
print(res)

# How to find the common contexts of one or more words
res = text.common_contexts(["the", "moors"])
print(res)

# How to find how many times a word occurs in a text
print(text.count("Heathcliff"))

# How to calculate how often a word occurs in a text
count = text.count("Heathcliff") # How many times the word occurs
total = len(text) # Total number of words in the text
freq = count/total # Percentage of the text made up of that word

# How to analyze sentiments with polarity scores (positive, negative, neutral)
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
# Analyze the raw file (after it has been read and decoded, but before it has been tokenized)
scores = sid.polarity_scores(raw)
print(scores)

# How to open and read a file
file = open("wh.txt", "r")
book = file.read()

# How to get the polarity scores of the first 10 sentences in the file
book = nltk.sent_tokenize(book) # Tokenize the book into sentences
for i in range(10): # Repeat 10 times
    sentence_scores = sid.polarity_scores(book[i])
    print(sentence_scores)

# Open another file
file2 = open("katebush.txt", "r")
lyrics = file2.read()

# How to break up the file by newlines (line breaks) and store it as a list instead of a string
lyrics = lyrics.split("\n")

# How to get the polarity scores of each lyric
for lyric in lyrics:
    lyric_scores = sid.polarity_scores(lyric)
    print(lyric_scores)
