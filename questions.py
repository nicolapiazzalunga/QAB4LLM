import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


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
    content_dict = dict()
    for file_name in os.listdir(directory):
        with open(os.path.join(directory, file_name), encoding="utf-8") as content_file:
            content = content_file.read()
        content_dict[file_name] = content
    return content_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    filtered_document = list()
    document = document.lower()
    tokenized_document = nltk.word_tokenize(document)
    stop_words = nltk.corpus.stopwords.words("english")
    punctuation = string.punctuation
    for word in tokenized_document:
        if word in stop_words or word in punctuation:
            continue
        filtered_document.append(word)
    filtered_document.sort()
    return filtered_document


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words_occurrences = dict()
    idfs = dict()
    number_of_documents = len(documents)
    for document in documents:
        words_checked = list()
        for word in documents[document]:
            if word in words_occurrences:
                if word in words_checked:
                    continue
                words_occurrences[word] += 1
                words_checked.append(word)
            else:
                words_occurrences[word] = 1
                words_checked.append(word)
    for word in words_occurrences:
        idfs[word] = math.log(number_of_documents / words_occurrences[word])
    return idfs


def compute_tf(document):
    """
    Given a 'document' list of words, returns a dictionary containing 
    the term frequencies
    """
    term_frequency = dict()
    for word in document:
        if word in term_frequency:
            term_frequency[word] += 1
        else:
            term_frequency[word] = 1
    return term_frequency


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    ranking = list()
    for document in files:
        idf_tf = 0
        tfs = compute_tf(files[document])
        for word in query:
            if word in tfs:
                idf_tf += (idfs[word] * tfs[word])
        ranking.append((document, idf_tf))
        
    ranking.sort(reverse=True, key=lambda x: x[1])

    top_files = list()
    for i in range(n):
        top_files.append(ranking[i][0])

    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking = list()
    for sentence in sentences:
        matching_word_measure = 0
        query_term_density = 0
        tfs = compute_tf(sentences[sentence])
        for word in query:
            if word in sentences[sentence]:
                matching_word_measure += idfs[word]
                query_term_density += tfs[word]
        query_term_density /= len(sentences[sentence])
        ranking.append((sentence, matching_word_measure, query_term_density))
    
    ranking.sort(reverse=True, key=lambda x: x[2])
    ranking.sort(reverse=True, key=lambda x: x[1])

    top_sentences = list()
    for i in range(n):
        top_sentences.append(ranking[i][0])

    return top_sentences


if __name__ == "__main__":
    main()
