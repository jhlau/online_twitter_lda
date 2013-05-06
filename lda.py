#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.
# Updated by Jey Han Lau, 2012
# Updates:
# -Parallelisation of the Gibbs Sampler
# -Online Processing of Documents
# Related Paper:
# -On-line Trend Analysis with Topic Models: #twitter trends detection topic model online
#   (COLING 2012)

import numpy
import operator
import time
import os
import pickle
from multiprocessing import Pool
import threading

def parallel_inference(i, st, ed, o_docs, o_z_m_n, o_n_m_z, o_n_z_t, o_n_z):

#    print "i =", i, "st =", st, "ed =", ed, "docs =", o_docs
#    print "BEFORE:"
#    print "\tz_m_n =", o_z_m_n
#    print "\tn_m_z =", o_n_m_z
#    print "\tn_z_t =", o_n_z_t
#    print "\tn_z =", o_n_z

    for m, doc in enumerate(o_docs):
        z_n = o_z_m_n[m]
        n_m_z = o_n_m_z[m]
        for n, t in enumerate(doc):
            # discount for n-th word t with topic z
            z = z_n[n]
            n_m_z[z] -= 1
            o_n_z_t[z, t] -= 1
            o_n_z[z] -= 1
            

            # sampling topic new_z for t
            p_z = o_n_z_t[:, t] * n_m_z / o_n_z
            new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()

            # set z the new topic and increment counters
            z_n[n] = new_z
            n_m_z[new_z] += 1
            o_n_z_t[new_z, t] += 1
            o_n_z[new_z] += 1


#    print "AFTER:"
#    print "\tz_m_n =", o_z_m_n
#    print "\tn_m_z =", o_n_m_z
#    print "\tn_z_t =", o_n_z_t
#    print "\tn_z =", o_n_z

    return (i, st, ed, o_z_m_n, o_n_m_z, o_n_z_t, o_n_z)

class LDA:
    def __init__(self, K, alpha, beta, cont, docs, docs_nt, V, docs_times, output_dir, prev_lda, \
        nproc, smartinit=True):
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.cont = cont # contribution proportion of history topics
        self.docs = docs
        self.docs_nt = docs_nt
        self.docs_times = docs_times
        self.V = V
        self.output_dir = output_dir
        self.nproc = nproc
        self.tlock = threading.Lock()

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = numpy.zeros((len(self.docs), K)) + alpha
        self.n_z_t = numpy.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_z_t_new = numpy.zeros((K, V)) # new word count of each topic and vocabulary
        self.n_z = numpy.zeros(K) + V * beta    # word count of each topic
        if prev_lda != None:
            #convert the old model's topic-word matrix counts to proportion
            sum_n_z_t = 0
            for z_t in prev_lda.n_z_t:
                sum_n_z_t += sum(z_t)
            for (z, z_t) in enumerate(prev_lda.n_z_t):
                for (t, val) in enumerate(z_t):
                    self.n_z_t[z, t] = ((float(prev_lda.n_z_t[z,t]) / \
                        sum_n_z_t)*self.V*self.K*self.beta*(self.cont)) \
                        + (self.beta*(1.0-self.cont))
            for (z, val) in enumerate(self.n_z):
                self.n_z[z] = sum(self.n_z_t[z])

            for (row_id, row) in enumerate(prev_lda.n_m_z):
                for (col_id, col) in enumerate(row):
                    self.n_m_z[row_id][col_id] = col

        self.N = 0

        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = numpy.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(numpy.array(z_n))

            #update the document timestamp
            prev_time = docs_times[m]

    def parallel_inference_complete(self, result):
        (i, st, ed, z_m_n, n_m_z, n_z_t, n_z) = result

        self.tlock.acquire()
        #update z_m_n and n_m_z
        self.z_m_n[st:ed] = z_m_n
        self.n_m_z[st:ed] = n_m_z

        #update n_z_t (reduce-scatter operation)
        self.n_z_t_new = self.n_z_t_new + (n_z_t - self.n_z_t)

        self.tlock.release()
        

    def inference(self):
#        print "ORIGINAL:"
#        print "\tdocs =", self.docs
#        print "\tz_m_n =", self.z_m_n
#        print "\tn_m_z =", self.n_m_z
#        print "\tn_z_t =", self.n_z_t
#        print "\tn_z =", self.n_z
        #refesh the n_z_t array used for storing new counts
        self.n_z_t_new = numpy.zeros((self.K, self.V))
        #Spawn a number of threads to do the inference
        po = Pool()
        num_doc_per_proc = float(len(self.docs))/self.nproc
        for i in range(0, self.nproc):
            st = int(round(float(i)*num_doc_per_proc))
            ed = int(round(float(i+1)*num_doc_per_proc))
            po.apply_async(parallel_inference, \
                (i, st, ed, self.docs[st:ed], self.z_m_n[st:ed], self.n_m_z[st:ed], \
                self.n_z_t, self.n_z), callback=self.parallel_inference_complete)
    
        po.close()
        po.join()

        #update n_z_t
        self.n_z_t = self.n_z_t + self.n_z_t_new
        #update n_z
        self.n_z = numpy.sum(self.n_z_t, 1)

#        print "MERGED:"
#        print "\tz_m_n =", self.z_m_n
#        print "\tn_m_z =", self.n_m_z
#        print "\tn_z_t =", self.n_z_t
#        print "\tn_z =", self.n_z

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:,w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)

def drop_one_day_data(voca, lda, alpha):
    n_firstday = lda.docs_nt[0]
    #decrement the counts of topic-word matrix for the documents to be removed
    for (m, doc) in enumerate(lda.docs[:n_firstday]):
        for (n, t) in enumerate(doc):
            z = lda.z_m_n[m][n]
            lda.n_z_t[z, t] -= 1
            lda.n_z[z] -= 1
            voca.wordfreq[t] -= 1

    lda.docs = lda.docs[n_firstday:]
    lda.docs_nt = lda.docs_nt[1:]
    lda.docs_times = lda.docs_times[n_firstday:]
    lda.z_m_n = lda.z_m_n[n_firstday:]
    lda.n_m_z = lda.n_m_z[n_firstday:]

    #convert the n_m_z counts to priors
    total_n_m_z = 0.0
    for n_m in lda.n_m_z:
        total_n_m_z += sum(n_m)
    for (m, n_m) in enumerate(lda.n_m_z):
        for (z, count) in enumerate(n_m):
            new_prior = (float(count)/(total_n_m_z))*len(lda.n_m_z)*lda.K*alpha
            lda.n_m_z[m][z] = new_prior

    return (voca, lda)

def lda_learning(lda, iteration, voca):
    pre_perp = lda.perplexity()
    print "initial perplexity=%f" % pre_perp
    for i in range(iteration):
        start = time.time()
        lda.inference()
        print "(%.1fs) iter=%d" % (time.time()-start, i + 1),
        if ( (i+1)%50 == 0):
            perp = lda.perplexity()
            print "p=%f" % (perp)
        else:
            print
    output_word_topic_dist(lda, voca)

def output_word_topic_dist(lda, voca):
    phi = lda.worddist()
    topics_file = open(lda.output_dir + "/topics.txt", "w")
    for k in range(lda.K):
        #print "\n-- topic: %d" % k
        for w in numpy.argsort(-phi[k])[:10]:
            #print "%s: %f" % (voca[w], phi[k,w])
            topics_file.write(voca[w] + " ")
        topics_file.write("\n")

def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", type="string", help="corpus filename")
    parser.add_option("-t", dest="time_file", help="timestamp of documents")
    parser.add_option("-o", dest="output_dir", type="string", help="output directory")
    parser.add_option("-m", dest="model", help="previously trained model")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.01)
    parser.add_option("-p", dest="cont", type="float", help="parameter contribution proportion", \
        default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=50)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=500)
    parser.add_option("-s", dest="smartinit", action="store_false", \
        help="smart initialize of parameters", default=True)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", \
        action="store_true", default=True)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--wf", dest="wf", type="int", \
        help="threshold of word frequency to cut words", default=1)
    parser.add_option("--num-proc", dest="nproc", type="int", help="number of processors", \
        default=4)
    (options, args) = parser.parse_args()
    if not (options.filename) or \
        not (options.time_file) or not(options.output_dir):
        parser.error("need (corpus filename(-f) and "
            "document timestamp file(-t) and output directory(-o)")

    if options.filename:
        corpus = vocabulary.load_file(options.filename)
    if options.seed != None:
        numpy.random.seed(options.seed)

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    
    voca = vocabulary.Vocabulary(options.stopwords, options.wf)

    if options.model:
        (prev_voca, prev_lda) = pickle.load(open(options.model))
        #drop one day worth's of data to accommodate the new day's data
        prev_voca, prev_lda = drop_one_day_data(prev_voca, prev_lda, options.alpha)
        options.K = prev_lda.K
    else:
        prev_lda = None
        prev_voca = None

    #generate the vocabularies for voca
    voca.gen_vocabs(corpus, prev_voca, prev_lda)

    docs = [voca.doc_to_ids(doc) for doc in corpus]
    #calculate the number of elements for each timestamp group in docs
    docs_nt = []
    docs_times = [ item.strip() for item in open(options.time_file).readlines() ]
    tmp_nt = {}
    for time in set(docs_times):
        tmp_nt[time] = docs_times.count(time)
    for (time, count) in sorted(tmp_nt.items()):
        docs_nt.append(count)
    tmp_nt.clear()

    if options.model:
        #update docs
        tmp_docs = []
        tmp_docs.extend(prev_lda.docs)
        tmp_docs.extend(docs)
        docs = tmp_docs

        #update docs_times
        tmp_docs_times = []
        tmp_docs_times.extend(prev_lda.docs_times)
        tmp_docs_times.extend(docs_times)
        docs_times = tmp_docs_times

        #update docs_nt
        tmp_docs_nt = []
        tmp_docs_nt.extend(prev_lda.docs_nt)
        tmp_docs_nt.extend(docs_nt)
        docs_nt = tmp_docs_nt

    #if options.wf > 0: docs = voca.cut_low_freq(docs, options.wf)

    #initialise lda
    lda = LDA(options.K, options.alpha, options.beta, options.cont, docs, docs_nt, voca.size(), \
        docs_times, options.output_dir, prev_lda, options.nproc, options.smartinit)

    #print word frequency
    freqword = {}
    freqword_file = open(lda.output_dir + "/freqwords.txt", "w")
    for (vocab_id, freq) in enumerate(voca.wordfreq):
        freqword[voca.vocas[vocab_id]] = freq
    for (vocab, freq) in sorted(freqword.items(), key=operator.itemgetter(1), reverse=True):
        freqword_file.write(vocab + " " + str(freq) + "\n")
    freqword_file.flush()

    print "corpus=%d, words=%d, K=%d, a=%f, b=%f, nproc=%d" % (len(corpus), len(voca.vocas), 
options.K, options.alpha, options.beta, options.nproc)

    #import cProfile
    #cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    lda_learning(lda, options.iteration, voca)

    #save the model for potential re-use later
    lda.tlock = None
    pickle.dump((voca, lda), open(options.output_dir + "/model.dat", "w"))

if __name__ == "__main__":
    main()
