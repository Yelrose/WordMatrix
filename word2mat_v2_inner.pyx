#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

from libc.stdio cimport printf

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)


try:
    from scipy.linalg.blas import fblas
except:
    import scipy.linalg.blas as fblas


REAL = np.float32
ctypedef np.float32_t REAL_t

DEF MAX_SENTENCE_LEN = 10000



ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil
ctypedef void (*sgemv_ptr) (const char* TRANS,const int *M,const int *N,const float*  alpha,const float *A,const int *LDA,const float * X,const int *incX,const float *beta,float * Y,const int * incY) nogil
ctypedef void (*sger_ptr) (const int * M,const int * N,const float * alpha,const float *X,const int *incX,float * Y,const int* incY,const float * A,const int* LDA) nogil
ctypedef REAL_t (*our_dot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef void (*our_saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil

cdef our_dot_ptr our_dot
cdef our_saxpy_ptr our_saxpy





cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
cdef sgemv_ptr sgemv=<sgemv_ptr>PyCObject_AsVoidPtr(fblas.sgemv._cpointer)
cdef sger_ptr sger = <sger_ptr>PyCObject_AsVoidPtr(fblas.sger._cpointer)

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]


cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int vector_size,const int topic_size,
    const REAL_t * context_vector,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,REAL_t * neu1, REAL_t *word_locks) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * vector_size * topic_size, row2
    cdef REAL_t f, g
    cdef char trans = <char >'c'
    memset(neu1,0,vector_size*cython.sizeof(REAL_t))
    memset(work, 0, vector_size * cython.sizeof(REAL_t))
    sgemv(&trans,&topic_size,&vector_size,&ONEF,&syn0[row1],&topic_size,context_vector,&ONE,&ONEF,neu1,&ONE)
    for b in range(codelen):
        row2 = word_point[b] *vector_size
        f = our_dot(&vector_size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&vector_size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&vector_size, &g, neu1, &ONE, &syn1[row2], &ONE)
    sger(&topic_size,&vector_size,&ONEF,context_vector,&ONE,work,&ONE,&syn0[row1],&topic_size)


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random

cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int vector_size,const int topic_size,REAL_t* context_vector, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,REAL_t *neu1,
    unsigned long long next_random, REAL_t *word_locks) nogil:

    cdef long long a
    cdef long long row1 = word2_index * vector_size*topic_size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, vector_size * cython.sizeof(REAL_t))
    memset(neu1, 0,  vector_size * cython.sizeof(REAL_t))
    cdef char trans = <char> 'c'
    sgemv(&trans,&topic_size,&vector_size,&ONEF,&syn0[row1],&topic_size,context_vector,&ONE,&ONEF,neu1,&ONE)
    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * vector_size

        f = our_dot(&vector_size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&vector_size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&vector_size, &g, neu1, &ONE, &syn1neg[row2], &ONE)
    sger(&topic_size,&vector_size,&ONEF,context_vector,&ONE,work,&ONE,&syn0[row1],&topic_size)

    return next_random


cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count = 1.0
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha
        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j, k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m] * size], &ONE)


cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random



def train_sentence_sg(model, sentence,topic_vector, alpha, _work,_neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *context_vector= <REAL_t *>(np.PyArray_DATA(topic_vector))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work= <REAL_t *>(np.PyArray_DATA(_work))
    cdef REAL_t *neu1= <REAL_t *>(np.PyArray_DATA(_neu1))
    cdef REAL_t _alpha = alpha
    cdef int vector_size = model.vector_size
    cdef int topic_size = model.topic_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int topic[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window
    cdef int topic_window = model.topic_window

    cdef int i, j, k
    cdef int tot_topic
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)


    vlookup = model.vocab
    i = 0
    for token,topic_id in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # leaving i unchanged/shortening sentence
        if sample and word.sample_int < random_int32(&next_random):
            continue
        indexes[i] = word.index
        topic[i] = <int > topic_id
        if hs:
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # single randint() call avoids a big thread-sync slowdown
    for i, item in enumerate(model.random.randint(0, window, sentence_len)):
        reduced_windows[i] = item

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            memset(context_vector, 0, topic_size * cython.sizeof(REAL_t))
            j = i - topic_window
            if j < 0:
                j = 0
            k = i + topic_window + 1
            tot_topic =  k - j
            if k > sentence_len:
                k =  sentence_len
                for j in range(j,k):
                    context_vector[topic[j]] += 1.0 / tot_topic

            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i:
                    continue
                if hs:
                    fast_sentence_sg_hs(points[j], codes[j], codelens[j], syn0, syn1, vector_size,topic_size,context_vector, indexes[i], _alpha, work,neu1, word_locks)
                if negative:
                    next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn0, syn1neg, vector_size,topic_size,context_vector, indexes[j], indexes[i], _alpha, work,neu1, next_random, word_locks)

    return result


def train_sentence_cbow(model, sentence, alpha, _work, _neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.syn0_lockf))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef REAL_t _alpha = alpha
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative or frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.cum_table))
        cum_table_len = len(model.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    vlookup = model.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # leaving i unchanged/shortening sentence
        if sample and word.sample_int < random_int32(&next_random):
            continue
        indexes[i] = word.index
        if hs:
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # single randint() call avoids a big thread-sync slowdown
    for i, item in enumerate(model.random.randint(0, window, sentence_len)):
        reduced_windows[i] = item

    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            if hs:
                fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks)
            if negative:
                next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks)

    return result


# Score is only implemented for hierarchical softmax
def score_sentence_sg(model, sentence, _work):

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))
    
    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            result += 1

    # release GIL & train on the sentence
    work[0] = 0.0

    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window 
            if j < 0:
                j = 0
            k = i + window + 1 
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                score_pair_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], work)

    return work[0]

cdef void score_pair_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, REAL_t *work) nogil:

    cdef long long b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f

    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f = sgn*f
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f

def score_sentence_cbow(model, sentence, _work, _neu1):

    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef int size = model.layer1_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
        word = sentence[i]
        if word is None:
            codelens[i] = 0
        else:
            indexes[i] = word.index
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
            result += 1

    # release GIL & train on the sentence
    work[0] = 0.0
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window 
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            score_pair_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, work, i, j, k, cbow_mean)

    return work[0]

cdef void score_pair_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count, sgn
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f = sgn*f
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy 
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
