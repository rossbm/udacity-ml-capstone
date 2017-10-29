import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import wordpunct_tokenize
from nltk import sent_tokenize

def create_embedmatrix(embedding_file, word_index):
    #word embedding
    embeddings_index = {}
    not_found = {}
    f = open(embedding_file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_dim = next(iter(embeddings_index.values())).shape[0]
    
    #now make embedding
    embedding_matrix = np.zeros((len(word_index), embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found[word] = not_found.get(word, 0) + 1   
    return embedding_matrix, embedding_dim, not_found

def create_seqs(tokens, vocab, MAX_LEN):
    seqs = np.zeros((len(tokens), MAX_LEN), dtype=np.int32)

    #truncate trailing, pad at end  (since simpler, can try ecpirmeting later)
    for i, joke in enumerate(tokens):
        for j, word in enumerate(joke):
            if j >= MAX_LEN:
                break
            seqs[i, j] = vocab[word]
    return seqs

def create_vocab(texts):
    punkt_tokens = CountVectorizer(tokenizer=wordpunct_tokenize, lowercase=True)
    punkt_tokens.fit(texts)
    vocab = punkt_tokens.vocabulary_
    return vocab

def create_sent_seqs(text):
    sents = sent_tokenize(text)
    tokens = list()
    for sent in sents:
        tokens.append(wordpunct_tokenize(sent.lower()))
    return tokens

def create_sent(texts, vocab):
    seqs_list = []
    for text in texts:
        seqs_list.append(create_sent_seqs(text))
    return seqs_list