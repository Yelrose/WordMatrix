#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division  # py3 "true division"
import logging
import sys
import os
import heapq
from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import time

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models import ldamodel
from gensim.corpora import dictionary
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType

logger = logging.getLogger("word2mat")

try:
    from word2mat_inner import train_sentence_sg, train_sentence_cbow, FAST_VERSION
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_sentence_sg(model, sentence, context_vector,alpha, work=None,neu1=None):
        """
        Update skip-gram model by training on a single sentence.

        The sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `word2mat.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2mat_inner instead.

        """
        word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                       model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2mat code

            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - model.window + reduced_window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                # don't train on the `word` itself
                if pos2 != pos:
                    train_sg_pair(model, model.index2word[word.index], word2.index,context_vector, alpha)

        return len(word_vocabs)

    def train_sentence_cbow(model, sentence,context_vector, alpha, work=None, neu1=None):
        """
        Update CBOW model by training on a single sentence.

        The sentence is a list of string tokens, which are looked up in the model's
        vocab dictionary. Called internally from `word2mat.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2mat_inner instead.

        """
        word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                       model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2mat code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.syn0[word2_indices], axis=0)  # 1 x vector_size
            l1 = l1.reshape(model.topic_size,model.vector_size)
            l1 = l1.T.dot(context_vector)
            if word2_indices and model.cbow_mean:
                l1 /= len(word2_indices)
            train_cbow_pair(model, word, word2_indices,context_vector, l1, alpha)

        return len(word_vocabs)


def train_sg_pair(model, word, context_index,topic_vector, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):
    if context_vectors is None:
        context_vectors = model.syn0
    if context_locks is None:
        context_locks = model.syn0_lockf

    if word not in model.vocab:
        return
    predict_word = model.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index].reshape(model.vector_size,model.topic_size)  # input word (NN input/projection layer)
    l1 = l1.dot(topic_vector)

    lock_factor = context_locks[context_index]

    neu1e = zeros(l1.shape)

    if model.hs:
        # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
        l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
        fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [predict_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        context_vectors[context_index] += lock_factor *outer(neu1e,topic_vector).reshape(model.vector_size*model.topic_size)
    return neu1e


def train_cbow_pair(model, word, input_word_indices, topic_vector,l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)

    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        fa = 1. / (1. + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        for i in input_word_indices:
            model.syn0[i] += outer(topic_vector,neu1e).reshape(model.vector_size*model.topic_size) * model.syn0_lockf[i]
    return neu1e

class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class Word2Mat(utils.SaveLoad):
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2mat/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2mat implementation via `save_word2mat_format()` and `load_word2mat_format()`.

    """
    def __init__(
            self, sentences=None,sentences_vector=None, size=100,topic=50, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001,
            sg=1, hs=1, negative=0, cbow_mean=0, hashfxn=hash, iter=1, null_word=0,other_model=None):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=1`), skip-gram is used.
        Otherwise, `cbow` is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).

        `min_count` = ignore all words with total frequency lower than this.

        `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types
        need about 1GB of RAM. Set to `None` for no limit (default).

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 0 (off), useful value is 1e-5.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0).

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).

        `cbow_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when cbow is used.

        `hashfxn` = hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        `iter` = number of iterations (epochs) over the corpus.

        """
        self.sentences_vector = sentences_vector
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.topic_size =int(topic)
        self.other_model = other_model
        #self.lda_iter= int(lda_iter)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 0
        self.total_train_time = 0
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")

            self.build_vocab(sentences)
            #self.train_lda(sentences)
            self.train(sentences)

    def train_lda(self,sentences):
        '''
            train lda_model
        '''
        self.lda_dic = dictionary.Dictionary(sentences)
        self.ldamodel = ldamodel.LdaModel(BagOfWordSentences(sentences,self.lda_dic),num_topics=self.topic_size,passes=self.lda_iter)




    def make_cum_table(self, power=0.75, domain=2**31 - 1):
        """
        Create a cumulative-distribution table using stored vocabulary word counts for
        drawing random words in the negative-sampling training routines.

        To draw a word index, choose a random integer up to the maximum value in the
        table (cum_table[-1]), then finding that integer's sorted insertion point
        (as if by bisect_left or ndarray.searchsorted()). That insertion point is the
        drawn index, coming up in proportion equal to the increment at that slot.

        Called internally from 'build_vocab()'.
        """
        vocab_size = len(self.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.vocab[word].count**power for word in self.vocab]))
        cumulative = 0.0
        for word_index in range(vocab_size):
            cumulative += self.vocab[self.index2word[word_index]].count**power / train_words_pow
            self.cum_table[word_index] = round(cumulative * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words", len(self.vocab))

        # build the huffman tree
        heap = list(itervalues(self.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i", max_depth)

    def build_vocab(self, sentences, keep_raw_vocab=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences)  # initial survey
        self.scale_vocab(keep_raw_vocab)  # trim by min_count & precalculate downsampling
        self.finalize_vocab()  # build tables & arrays

    def scan_vocab(self, sentences, progress_per=10000):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all words and their counts")
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words, keeping %i word types",
                            sentence_no, sum(itervalues(vocab)) + total_words, len(vocab))
            for word in sentence:
                vocab[word] += 1

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                total_words += utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab

    def scale_vocab(self, min_count=None, sample=None, dry_run=False, keep_raw_vocab=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        min_count = min_count or self.min_count
        sample = sample or self.sample

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            self.vocab = {}
        drop_unique, drop_total, retain_total, original_total = 0, 0, 0, 0
        retain_words = []
        for word, v in iteritems(self.raw_vocab):
            if v >= min_count:
                retain_words.append(word)
                retain_total += v
                original_total += v
                if not dry_run:
                    self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                    self.index2word.append(word)
            else:
                drop_unique += 1
                drop_total += v
                original_total += v
        logger.info("min_count=%d retains %i unique words (drops %i)",
                    min_count, len(retain_words), drop_unique)
        logger.info("min_count leaves %i word corpus (%i%% of original %i)",
                    retain_total, retain_total * 100 / max(original_total, 1), original_total)

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items", len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words", sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # return from each step: words-affected, resulting-corpus-size
        report_values = {'drop_unique': drop_unique, 'retain_total': retain_total,
                         'downsample_unique': downsample_unique, 'downsample_total': int(downsample_total)}


        return report_values

    def finalize_vocab(self):
        """Build tables and model weights based on final vocabulary settings."""
        if not self.index2word:
            self.scale_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            word, v = '\0', Vocab(count=1, sample_int=0)
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
        # set initial input/projection and hidden weights
        self.reset_weights()


    def reset_weights_from_model(self):
        other_model = self.other_model
        logger.info("resetting layer weights from other")
        for key,word in self.vocab.items():
            idx = word.index
            if key in other_model:
                vec0 = other_model[key]
                vec = zeros((self.topic_size,self.vector_size),dtype=REAL)
                for i in xrange(self.topic_size):
                    vec[i] = vec0
                self.syn0[idx] = vec.T.reshape(self.topic_size * self.vector_size)
            else : self.syn0[idx] = self.seeded_vector(self.index2word[idx] + str(self.seed))










    def _do_train_job(self, job, alpha, inits):

        work, neu1 = inits
        tally = 0
        raw_tally = 0
        for sentence_id,sentence in job:
            context_vector = self.sentences_vector[sentence_id]
            if self.sg:
                tally += train_sentence_sg(self, sentence,context_vector, alpha, work,neu1)
            else:
                tally += train_sentence_cbow(self, sentence,context_vector, alpha, work, neu1)
            raw_tally += len(sentence)
        return (tally, raw_tally)

    def _raw_word_count(self, items):
        return sum(len(item) for item in items)

    def train(self, sentences, total_words=None, word_count=0, chunksize=100, total_examples=None, queue_factor=2, report_delay=1):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For word2mat, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for word2mat, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s and negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not hasattr(self, 'syn0'):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i examples, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")
        logging.info("initiallize sentence")
        sentences = EnumerateSentence(sentences)
        logging.info("initiallize sentence finish")
        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_init():
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            return (work, neu1)

        def worker_one_job(job, inits):
            items, alpha = job
            if items is None:  # signal to finish
                return False
            # train & return tally
            tally, raw_tally = self._do_train_job(items, alpha, inits)
            progress_queue.put((len(items), tally, raw_tally))  # report progress
            return True

        def worker_loop():
            """Train the model, lifting lists of sentences from the jobs queue."""
            init = worker_init()
            while True:
                job = job_queue.get()
                if not worker_one_job(job, init):
                    break

        start, next_report = default_timer(), 1.0

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        if self.workers > 0:
            job_queue = Queue(maxsize=queue_factor * self.workers)
        else:
            job_queue = FakeJobQueue(worker_init, worker_one_job)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        pushed_words = 0
        pushed_examples = 0
        example_count = 0
        trained_word_count = 0
        raw_word_count = word_count
        push_done = False
        done_jobs = 0
        next_alpha = self.alpha
        jobs_source = enumerate(utils.grouper(sentences, chunksize))
        # fill jobs queue with (sentence, alpha) job tuples
        while True:
            try:
                job_no, items = next(jobs_source)
                logger.debug("putting job #%i in the queue at alpha %.05f", job_no, next_alpha)
                job_queue.put((items, next_alpha))
                # update the learning rate before every next job
                if self.min_alpha < next_alpha:
                    if total_examples:
                        # examples-based decay
                        pushed_examples += len(items)
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * (pushed_examples / total_examples)
                    else:
                        # words-based decay
                        #pushed_words += self._raw_word_count(items)
                        pushed_words += self._raw_word_count([item[1] for item in items])
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * (pushed_words / total_words)
                    next_alpha = max(next_alpha, self.min_alpha)
            except StopIteration:
                logger.info(
                    "reached end of input; waiting to finish %i outstanding jobs",
                    job_no - done_jobs + 1)
                for _ in xrange(self.workers):
                    job_queue.put((None, 0))  # give the workers heads up that they can finish -- no more work!
                push_done = True
            try:
                while done_jobs < (job_no+1) or not push_done:
                    examples, trained_words, raw_words = progress_queue.get(push_done)  # only block after all jobs pushed
                    example_count += examples
                    trained_word_count += trained_words  # only words in vocab & sampled
                    raw_word_count += raw_words
                    done_jobs += 1
                    elapsed = default_timer() - start
                    if elapsed >= next_report:
                        if total_examples:
                            # examples-based progress %
                            logger.info(
                                "PROGRESS: at %.2f%% examples, %.0f words/s",
                                100.0 * example_count / total_examples, trained_word_count / elapsed)
                        else:
                            # words-based progress %
                            logger.info(
                                "PROGRESS: at %.2f%% words, %.0f words/s",
                                100.0 * raw_word_count / total_words, trained_word_count / elapsed)
                        next_report = elapsed + report_delay  # don't flood log, wait report_delay seconds
                else:
                    # loop ended by job count; really done
                    break
            except Empty:
                pass  # already out of loop; continue to next push

        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words took %.1fs, %.0f trained words/s",
            raw_word_count, elapsed, trained_word_count / elapsed if elapsed else 0.0)

        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()
        return trained_word_count

    def clear_sims(self):
        self.syn0norm = None

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.vector_size*self.topic_size), dtype=REAL)
        if self.other_model ==  None:
            # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
            for i in xrange(len(self.vocab)):
                # construct deterministic seed from word AND seed argument
                self.syn0[i] = self.seeded_vector(self.index2word[i] + str(self.seed))
        else :
            self.reset_weights_from_model()


        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        self.syn0norm = None

        self.syn0_lockf = ones(len(self.vocab), dtype=REAL)  # zeros suppress learning

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(uint32(self.hashfxn(seed_string)))
        return (once.rand(self.vector_size*self.topic_size) - 0.5) / self.vector_size

class FakeJobQueue(object):
    """Pretends to be a Queue; does equivalent of work_loop in calling thread."""
    def __init__(self, init_fn, job_fn):
        self.inits = init_fn()
        self.job_fn = job_fn

    def put(self, job):
        self.job_fn(job, self.inits)


class BrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in utils.smart_open(fname):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname, max_sentence_length=1000):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with utils.smart_open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split())  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
                words, rest = (utils.to_unicode(text[:last_token]).split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


class BagOfWordSentences(object):
    def  __init__(self,sentences,dictionary):
        self.sentences = sentences
        self.dictionary = dictionary
    def __iter__(self):
        for sentence in self.sentences:
            yield self.dictionary.doc2bow(sentence)


class LineSentence(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace."""
    def __init__(self, source, max_sentence_length=10000):
        """
        `source` can be either a string or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for line in self.source:
                line = utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i:(i + self.max_sentence_length)]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for line in fin:
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i:(i + self.max_sentence_length)]
                        i += self.max_sentence_length


class EnumerateSentence(object):
    def __init__(self,source):
        self.source =source

    def __iter__(self):
        for id,sentence in enumerate(self.source):
            yield (id,sentence)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sen = [['1','2','43','4','5','6',],['2','3','4','6','7','88','8',],['324','34','5','6','6','3']]
    sen_vector = [array([1,2,3,4]),array([2,2,3,4]),array([3,2,3,4])]
    model = Word2Mat(sen,sen_vector,topic=4,min_count=0,iter=5)
