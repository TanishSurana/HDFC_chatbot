import nltk
import sys
import os
import string
import math

from nltk.tokenize import word_tokenize

FILE_MATCHES = 2
SENTENCE_MATCHES = 2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dicc = dict()
    
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), encoding="utf8") as f:
            contents = f.read()
            dicc[filename] = contents
    
    return dicc


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = word_tokenize(document)
    litt = []
    common = nltk.corpus.stopwords.words("english")
    for word in words:
        ww = word.lower()
        ww = ww.translate(str.maketrans('', '', string.punctuation))
        if ww not in common and ww != "":
            litt.append(ww)

    return litt


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    dicc = dict()
    n = len(documents)
    for doc in documents:
        for word in documents[doc]:
            if word not in dicc:
                count = 0
                for d in documents:
                    if word in documents[d]:
                        count += 1
        
                idf = math.log(n/count)
                dicc[word] = idf  
    return dicc
    


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    dicc = dict.fromkeys(files, 0)

    for word in query:
        for doc in files:
            tf = files[doc].count(word)
            dicc[doc] += tf*idfs[word]

    sorted_dicc = {k: v for k, v in sorted(dicc.items(), key=lambda item: item[1], reverse=True)}

    litt = list(sorted_dicc.keys())

    return litt


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    litt = []
    for sent in sentences:
        match = 0
        density = 0
        for word in query:
            if word in sentences[sent]:
                match += idfs[word]
                density += sentences[sent].count(word)
        
        density = density / len(sent)

        litt.append([sent, match, density])

    litt = sorted(litt, key=lambda x: (x[1], x[2]), reverse=True)[:n]
    litt2 = []
    for row in litt:
        litt2.append(row[0])
    return litt2

        

    

    raise NotImplementedError


if __name__ == "__main__":
    main()
