#!/usr/bin/python

import string

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

signature_words = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """

    stemmer = SnowballStemmer("english")

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()
    # print all_text
    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        text_string = text_string.lower()
        for word in signature_words:
            text_string = text_string.replace(word, '')

        # project part 2: comment out the line below
        # words = text_string

        # split the text string into individual words, stem each word,
        # and append the stemmed word to words (make sure there's a single
        # space between each stemmed word)
        stop_words = stopwords.words("english")
        tokens = word_tokenize(text_string, 'english', True)
        tokens = [word for word in tokens if word not in stop_words]
        word_list = [stemmer.stem(token) for token in tokens]
        words = ' '.join(word_list)
    return words


def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text


if __name__ == '__main__':
    main()
