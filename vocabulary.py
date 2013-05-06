#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# Updated by Jey Han Lau, 2012
# Updates:
# -Parallelisation of the Gibbs Sampler
# -Online Processing of Documents
# Related Paper:
# -On-line Trend Analysis with Topic Models: #twitter trends detection topic model online
#   (COLING 2012)

import re
import numpy
import pickle
import operator

#def load_corpus(range):
#    m = re.match(r'(\d+):(\d+)$', range)
#    if m:
#        start = int(m.group(1))
#        end = int(m.group(2))
#        from nltk.corpus import brown as corpus
#        return [corpus.words(fileid) for fileid in corpus.fileids()[start:end]]

def load_file(filename):
    corpus = []
    f = open(filename, 'r')
    for line in f:
        #doc = re.findall(r'\w+(?:\'\w+)?',line)
        doc = []
        hashtag = False
        for word in line.strip().split():
            #remove @reply tweets
            #if (not word.startswith("@")) and (not word.startswith("#")):
            if word == "#":
                hashtag = True
            else:
                if hashtag:
                    doc.append("#" + word)
                    hashtag = False
                else:
                    doc.append(word)
        
        corpus.append(doc)
    f.close()
    return corpus

#stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list = [ item.strip() for item in open("stopwords.txt").readlines() ]
#recover_list = {"wa":"was", "ha":"has"}
#wl = nltk.WordNetLemmatizer()

def is_stopword(w):
    return w in stopwords_list
#def lemmatize(w0):
#    w = wl.lemmatize(w0.lower())
#    #if w=='de': print w0, w
#    if w in recover_list: return recover_list[w]
#    return w

class Vocabulary:
    def __init__(self, excluds_stopwords=True, wordfreq_threshold=10):
        #if to word
        self.vocas = [] #id to word
        self.vocas_id = dict() # word to id
        self.wordfreq = [] #word counts
        self.excluds_stopwords = excluds_stopwords
        self.wordfreq_threshold = wordfreq_threshold

    def gen_vocabs(self, corpus, prev_voca, prev_lda):
        #temporary word frequency dictionary
        tmp_wf = {}

        #case that there is not previous model
        if (prev_voca == None) and (prev_lda == None):
            for doc in corpus:
                for word in doc:
                    if (self.excluds_stopwords and is_stopword(word)) or (len(word)<3):
                        pass
                    else:
                        if word not in tmp_wf:
                            tmp_wf[word] = 0
                        tmp_wf[word] += 1

            #remove words below the threshold
            for (word,freq) in tmp_wf.items():
                if freq < self.wordfreq_threshold:
                    del tmp_wf[word]

            self.vocas = tmp_wf.keys()
            for (vid, word) in enumerate(self.vocas):
                self.vocas_id[word] = vid

            #initialise the actual wordfreq dictionary
            self.wordfreq = [0]*len(self.vocas) # id to document frequency
 
        #case that there is a previous model
        else:
            for doc in corpus:
                for word in doc:
                    if (self.excluds_stopwords and is_stopword(word)) or (len(word)<3):
                        pass
                    else:
                        if word in prev_voca.vocas_id:
                            prev_voca.wordfreq[prev_voca.vocas_id[word]] += 1
                        else:
                            if word not in tmp_wf:
                                tmp_wf[word] = 0
                            tmp_wf[word] += 1

            wordids_to_delete = []
            for wordid, freq in enumerate(prev_voca.wordfreq):
                if freq < self.wordfreq_threshold:
                    wordids_to_delete.append(wordid)
            #filter low frequency words for the temporary wordfreq dic
            for word, freq in tmp_wf.items():
                if freq < self.wordfreq_threshold:
                    del tmp_wf[word]
            #generate the new vocas
            for (wid, word) in enumerate(prev_voca.vocas):
                if wid not in wordids_to_delete:
                    self.vocas.append(word)
            for word in tmp_wf.keys():
                self.vocas.append(word)
            for (wid, word) in enumerate(self.vocas):
                self.vocas_id[word] = wid

#            #update word frequency from old docs
#            for (wordid, freq) in enumerate(prev_voca.wordfreq):
#                if wordid not in wordids_to_delete:
#                    self.wordfreq[self.vocas_id[prev_voca.vocas[wordid]]] = freq

            #update prev_lda topic-word matrix
            for wordid in sorted(wordids_to_delete, reverse=True):
                prev_lda.n_z_t = numpy.delete(prev_lda.n_z_t, wordid, 1)
            smooth = numpy.amin(prev_lda.n_z_t)

            for i in range(0, len(tmp_wf.keys())):
                prev_lda.n_z_t = numpy.append(prev_lda.n_z_t, ([[smooth]]*prev_lda.K), axis=1)

            #initialise the actual wordfreq dictionary
            self.wordfreq = [0]*len(self.vocas) # id to document frequency

            #update the old document to the new word id
            for (docid, doc) in enumerate(prev_lda.docs):
                doc_in_word = [ prev_voca.vocas[wordid] for wordid in doc ]
                new_doc = self.doc_to_ids(doc_in_word)
                prev_lda.docs[docid] = new_doc

        #clear the temporary wordfreq dictionary
        tmp_wf.clear()

    def term_to_id(self, term):
        if term in self.vocas_id:
            voca_id = self.vocas_id[term]
            return voca_id
        else:
            return None

    def doc_to_ids(self, doc):
        #print ' '.join(doc)
        list = []
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                list.append(id)
                self.wordfreq[id] += 1
        if "close" in dir(doc): doc.close()
        return list

#    def cut_low_freq(self, corpus, threshold=1):
#        new_vocas = []
#        new_wordfreq = []
#        self.vocas_id = dict()
#        conv_map = dict()
#        for id, term in enumerate(self.vocas):
#            freq = self.wordfreq[id]
#            if freq > threshold:
#                new_id = len(new_vocas)
#                self.vocas_id[term] = new_id
#                new_vocas.append(term)
#                new_wordfreq.append(freq)
#                conv_map[id] = new_id
#        self.vocas = new_vocas
#        self.wordfreq = new_wordfreq
#
#        def conv(doc):
#            new_doc = []
#            for id in doc:
#                if id in conv_map: new_doc.append(conv_map[id])
#            return new_doc
#        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in stopwords_list

