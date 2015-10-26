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
from libc.string cimport memset

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
ctypedef unsigned long long (*fast_sentence_sg_neg_ptr) (
    const int negative, np.uint32_t *table, unsigned long long table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random) nogil

ctypedef void (*fast_sentence_sg_hs_ptr)(
    const np.uint32_t * word_point,const np.uint8_t * word_code, const int codelen,
    REAL_t * syn0,const REAL_t * context_vector,REAL_t * syn1, const int vector_size,const int topic_size,
    const np.uint32_t word2_index,const REAL_t alpha,REAL_t * work,REAL_t * neu1) nogil


cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x
cdef sgemv_ptr sgemv=<sgemv_ptr>PyCObject_AsVoidPtr(fblas.sgemv._cpointer)
cdef sger_ptr sger = <sger_ptr>PyCObject_AsVoidPtr(fblas.sger._cpointer)
cdef fast_sentence_sg_hs_ptr fast_sentence_sg_hs

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef int ZERO = 0
cdef REAL_t ZEROF = <REAL_t>0.0


cdef void fast_sentence0_sg_hs(
    const np.uint32_t * word_point,const np.uint8_t * word_code, const int codelen,
    REAL_t * syn0,const REAL_t * context_vector,REAL_t * syn1, const int vector_size,const int topic_size,
    const np.uint32_t word2_index,const REAL_t alpha,REAL_t * work,REAL_t * neu1) nogil:
    cdef long long a,b
    cdef long long row1 =  word2_index * vector_size * topic_size
    cdef REAL_t f,g
    cdef char trans = <char> 't'
    memset(work,0,vector_size* cython.sizeof(REAL_t))
    sgemv(&trans,&topic_size,&vector_size,&ONEF,&syn0[row1],&topic_size,context_vector,&ONE,&ZEROF,neu1,&ONE)
    for b in range(codelen):
        row2 = word_point[b] * vector_size
        f = <REAL_t>dsdot(&vector_size,neu1,&ONE,&syn1[row2],&ONE)
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        g = (1 -  word_code[b] -f) * alpha
        saxpy(&vector_size,&g,&syn1[row2],&ONE,work,&ONE)
        saxpy(&vector_size,&g,neu1,&ONE,&syn1[row2],&ONE)
    sger(&topic_size,&vector_size,&ONEF,context_vector,&ONE,work,&ONE,&syn0[row1],&topic_size)

def train_sentence_sg(model, sentence, alpha,_context_vector, _work,_neu1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    '''
    syn0 = (vocabsz, vector_size * topic_size)
    syn1 = (vocbasz, vector_size)
    syn1neg = (vocbasz, vector_size)
    '''
    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.syn0))
    cdef REAL_t * context_vector = <REAL_t *> (np.PyArray_DATA(_context_vector))
    cdef REAL_t *work
    cdef REAL_t * neu1
    cdef REAL_t _alpha = alpha
    cdef int vector_size = model.vector_size
    cdef int topic_size =  model.topic_size

    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t word_index[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *table
    cdef unsigned long long table_len
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.syn1neg))
        table = <np.uint32_t *>(np.PyArray_DATA(model.vocab.table))
        table_len = len(model.vocab.table)
        next_random = (2**24) * np.random.randint(0, 2**24) + np.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)
    sentence_len = <int>min(MAX_SENTENCE_LEN, len(sentence))

    for i in range(sentence_len):
            #indexes[i] = word.index
        word_index[i] = sentence[i]
        reduced_windows[i] = np.random.randint(window)
        word = model.vocab.vocab[word_index[i]]
        if hs:
            codelens[i] = <int>len(word.code)
            codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
            points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        else:
            codelens[i] = 1
    # release GIL & train on the sentence
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window + reduced_windows[i]
            if j < 0:
                j = 0
            k = i + window + 1 - reduced_windows[i]
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                if hs:
                    fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, context_vector,syn1,vector_size, topic_size, word_index[j],  _alpha, work,neu1)
            #if negative:
                #    next_random = fast_sentence_sg_neg(negative, table, table_len, syn0, syn1neg, size, indexes[i], indexes[j], _alpha, work, next_random)
    return result

def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.

    """
    global fast_sentence_sg_hs
    global fast_sentence_sg_neg
    global fast_sentence_cbow_hs
    global fast_sentence_cbow_neg

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

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    fast_sentence_sg_hs =  fast_sentence0_sg_hs
    '''
    if (abs(d_res - expected) < 0.0001):
        fast_sentence_sg_hs = fast_sentence2_sg_hs
        fast_sentence_sg_neg = fast_sentence0_sg_neg
        fast_sentence_cbow_hs = fast_sentence0_cbow_hs
        fast_sentence_cbow_neg = fast_sentence0_cbow_neg
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        fast_sentence_sg_hs = fast_sentence2_sg_hs
        fast_sentence_sg_neg = fast_sentence1_sg_neg
        fast_sentence_cbow_hs = fast_sentence1_cbow_hs
        fast_sentence_cbow_neg = fast_sentence1_cbow_neg
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        fast_sentence_sg_hs = fast_sentence2_sg_hs
        fast_sentence_sg_neg = fast_sentence2_sg_neg
        fast_sentence_cbow_hs = fast_sentence2_cbow_hs
        fast_sentence_cbow_neg = fast_sentence2_cbow_neg
        return 2
    '''
    return 0

FAST_VERSION = init()  # initialize the module

