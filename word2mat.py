#-*- coding: utf-8 -*-
import heapq
import logging
import numpy as np
from copy import deepcopy
from gensim.models import ldamodel
from Queue import Queue, Empty
import threading
import timeit
import cPickle

def train_sg_sentence(model,sentence,alpha):
    '''
    train a sentence get the window
    '''
    tot_word = len(sentence)
    word_vocabs = [model.vocab.words[w] for w in sentence if w in model.vocab.words]
    word_vocabs = [w for w in word_vocabs if np.random.uniform(0,1) < 1 - np.sqrt(model.sampling/model.vocab.vocab[w].freq) ]
    lda = model.lda_model[model.vocab.get_bag_of_word(sentence)]
    context_vector = np.zeros(model.topic_size,dtype='float32')
    for topic,pro in lda:
        context_vector[topic] =  pro


    for pos,word in enumerate(word_vocabs):
        reduced_window = np.random.randint(model.window)
        start = max(0,pos - model.window + reduced_window)
        end = pos + model.window - reduced_window
        for pos2,word2 in enumerate(word_vocabs[start:end]):
            if pos2 == pos: continue
            train_sg_pair(model,word,word2,alpha,context_vector)
    return tot_word




def train_sg_pair(model,pos,target,alpha,context_vector):
    wordmat = model.syn0[pos].reshape(model.vector_size,model.topic_size)
    neu1 = wordmat.dot(context_vector)
    neu1e = np.zeros(neu1.shape)
    target_word = model.vocab.vocab[target]

    if model.hs:
        l2a = deepcopy(model.syn1[target_word.point])
        fa = 1.0 / (1.0 + np.exp(-np.dot(neu1,l2a.T)))
        ga = (1 - target_word.code - fa) * alpha
        model.syn1[target_word.point] += np.outer(ga,neu1)
        neu1e += np.dot(ga,l2a)

    if model.negative:
        word_indices = [target]
        while len(word_indices) < model.negative + 1:
            w = model.vocab.sampling()
            if w != target:
                word_indices.append(w)

        l2b = deepcopy(model.syn1neg[word_indices])
        fb = 1. / (1. + np.exp(-np.dot(neu1,l2b.T)))
        gb = (model.neg_labels - fb) * alpha
        model.syn1neg[word_indices] += np.outer(gb,neu1)
        neu1e += np.dot(gb,l2b)

    model.syn0[pos] += np.outer(neu1e,context_vector).reshape(model.vector_size*model.topic_size)


class Word:
    def __init__(self,word,cn,left=None,right=None):
        self.word = word
        self.code = []
        self.point = []
        self.cn = cn
        self.left=left
        self.right=right




class Vocab:
    def __init__(self):
        self.words = {}
        self.vocab = []
        self.vocab_sz = 0
        return


    def AddWord(self,word):
        if word not in self.words:
            self.vocab.append(Word(word,0))
            self.words[word] = self.vocab_sz
            self.vocab_sz += 1
        idx = self.words[word]
        self.vocab[idx].cn += 1

    def remove_infrequence(self,min_count):
        i = 0
        while i < self.vocab_sz:
            while i < self.vocab_sz and self.vocab[i].cn < min_count:
                self.vocab[i] = self.vocab[self.vocab_sz - 1]
                self.vocab_sz -= 1
            i += 1
        self.vocab = self.vocab[:self.vocab_sz]
        self.words = {}
        for i in xrange(self.vocab_sz):
            self.words[self.vocab[i].word] = i

    def build_table(self,sz = 100000000):
        self.table = np.zeros(sz,'int')
        tot = 0.
        for i in xrange(self.vocab_sz):
            tot += np.power(self.vocab[i].cn,0.75)

        j = 0
        cum = np.power(self.vocab[j].cn,0.75)/tot
        #print len(self.table)
        logging.info('Make negative sampling tables')
        for i in xrange(sz):
            self.table[i] = j
            if 1.0 * i / sz > cum:
                cum += np.power(self.vocab[j].cn,0.75)/tot
                j += 1
            if j == self.vocab_sz: j -= 1

    def sampling(self):
        return self.table[np.random.randint(0,len(self.table))]

    def build_tree(self):
        '''
            building binary tree for word2vec
        '''
        heap_sort = []
        for i in xrange(self.vocab_sz):
            heapq.heappush(heap_sort,(self.vocab[i].cn,i,-1,-1))

        for i in xrange(self.vocab_sz - 1):
            t1 = heapq.heappop(heap_sort)
            t2 = heapq.heappop(heap_sort)
            self.vocab.append(Word('_',t1[0]+t2[0],t1[-3],t2[-3]))
            heapq.heappush(heap_sort,(t1[0]+t2[0],i+self.vocab_sz,t1[-3],t2[-3]))

        for i in xrange(self.vocab_sz - 1):
            idx = 2* self.vocab_sz - 2 - i
            left = self.vocab[idx].left
            right = self.vocab[idx].right
            self.vocab[left].code = np.array(list(self.vocab[idx].code) + [0],dtype='uint8')
            self.vocab[right].code = np.array(list(self.vocab[idx].code) + [1],dtype='uint8')
            self.vocab[left].point = self.vocab[idx].point + [i]
            self.vocab[right].point = self.vocab[idx].point + [i]
        for i in xrange(self.vocab_sz):
            print i,self.vocab[i].code


    def get_frequence(self):
        sumf = 0.
        for i in xrange(self.vocab_sz):
            sumf += self.vocab[i].cn

        for i in xrange(self.vocab_sz):
            self.vocab[i].freq = self.vocab[i].cn / sumf

    def get_bag_of_word(self,sentence):
        bag_of_word = {}
        for wd in sentence:
            if wd in self.words:
                idx = self.words[wd]
                if idx not in bag_of_word:
                    bag_of_word[idx] = 0
                bag_of_word[idx] += 1
        return bag_of_word.items()



class Word2Mat:
    def __init__(self,sentences,size=100,topic_num=10,alpha=0.025,negative=5,sg=1,iter=3,hs=1,min_count=5,sampling=1e-3,workers=1,window=5):
        self.alpha = alpha
        self.negative = negative
        self.sg = sg
        self.iter = iter
        self.hs = hs
        self.min_count = min_count
        self.workers = workers
        self.vector_size = size
        self.topic_size = topic_num
        self.sentences = sentences
        self.sampling = sampling
        self.build_vocab()
        self.window = window
        logging.info('Total word %d'%(self.vocab.vocab_sz))
        self.topic_train()
        self.train()


    def build_vocab(self):
        logging.info('building vocab')
        self.vocab = Vocab()
        self.tot_word = 0
        for count,sentence in enumerate(self.sentences):
            if count % 10000 ==0 : logging.info('loading #%d sentence'% count)
            for word in sentence:
                self.tot_word += 1
                self.vocab.AddWord(word)
        self.vocab.get_frequence()
        self.vocab.remove_infrequence(self.min_count)
        if self.hs:
            self.vocab.build_tree()
        if self.negative > 0:
            self.vocab.build_table()

    def topic_train(self):
        logging.info('Training Topic Model')
        self.lda_model = ldamodel.LdaModel(corpus = BagOfWordSentences(self.sentences,self.vocab),num_topics=self.topic_size)



    def train(self):
        '''
            initilization of parameters
        '''
        layersz = self.vector_size * self.topic_size
        layer2sz = self.vector_size
        vocabsz = self.vocab.vocab_sz
        self.syn0 = np.array(np.random.uniform(-0.5/layersz,0.5/layersz,(vocabsz,layersz)),dtype='float32')
        self.syn1 = np.array(np.random.uniform(-0.5/layer2sz,0.5/layer2sz,(vocabsz,layer2sz)),dtype='float32')
        if self.negative > 0:
            self.syn1neg = np.array(np.random.uniform(-0.5/layer2sz,0.5/layer2sz,(vocabsz,layer2sz)),dtype='float32')
            self.neg_labels = np.zeros(self.negative + 1,dtype='uint8')
            self.neg_labels[0] = 1

        def worker_loop():
            '''
                get jobs in job queue
            '''
            while True:
                job = job_queue.get()
                item,alpha = job
                if item is None:
                    break
                word = 0
                if self.sg:
                    word += train_sg_sentence(self,sentence,alpha)
                else:
                        #word += train_sentence_cbow(self,sentence,alpha)
                    print 'train cbow'
                progress_queue.put(word)
        job_queue = Queue(maxsize=2* self.workers)
        progress_queue  = Queue(maxsize=3*self.workers)
        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        def progressor():
            start = timeit.default_timer()
            tot_word = self.tot_word * self.iter
            count_word = 0
            next_report = 0
            while True:
                try:
                    word = progress_queue.get(False)
                    count_word += word
                    elapsed = timeit.default_timer() - start
                    if elapsed >= next_report:
                        logging.info("PROGRESS: at %.2f%% ,%.0f words/s"%
                            (100.0*count_word/tot_word,count_word/elapsed))
                    next_report = elapsed + 1
                    if count_word > tot_word: break
                except Empty:
                    continue




        progress = threading.Thread(target=progressor)
        progress.daemon = True
        progress.start()
        for thread in workers:
            thread.daemon = True
            thread.start()

        count_word = 0
        tot_word = self.tot_word * self.iter
        for sentence in RepeatSentences(self.sentences,self.iter):
            alpha =self.alpha *(1 - 1.0 * count_word / tot_word)
            count_word += len(sentence)
            if alpha < self.alpha * 0.0001: alpha = self.alpha * 0.0001
            job_queue.put((sentence,alpha))
        for _ in xrange(self.workers):
            job_queue.put((None,0))

        for worker in workers:
            worker.join()
        progress.join()
        logging.info('Training finish')

    def save(self):
        cPickle.dump(self.syn0,open('./syn0.dmp','w'))
        cPickle.dump(self.vocab.vocab[:self.vocab.vocab_sz],open('./vocab.dmp','w'))
        cPickle.dump(self.lda_model,open('./lda_model.dmp','w'))






























class RepeatSentences:
    def __init__(self,sentences,iter):
        self.sentences = sentences
        self.iter = iter

    def __iter__(self):
        for i in xrange(self.iter):
            for sentence in sentences:
                yield sentence




class BagOfWordSentences:
    def __init__(self,sentences,vocab):
        self.vocab = vocab
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
           yield self.vocab.get_bag_of_word(sentence)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = [[1,2,3,3,4,5,5,5,555,5],[2,3,4],[5,5,6,77,7,7,8,9]]
    model = Word2Mat(sentences,iter=10,size=5,topic_num=2,min_count=0,workers=3)
    model.save()
    #lda_model = ldamodel.LdaModel(corpus = BagOfWordSentences(sentences,vocab),num_topics=3)



