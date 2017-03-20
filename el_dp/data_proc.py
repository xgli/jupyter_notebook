# coding: utf-8

import random
import numpy as np
import time
import tensorflow as tf
import input_data
import math
import gensim
import jieba
import os
import random
import cPickle as pickle
import time

#Parameters 
sen_fix_len = 20 #句子长度 
word_len = 300 #词向量长度
tr_size = 5000*2 #
batch_size = 40 
epoch_size = 100
learn_rate = 0.0001
display_step = 10

#Network parameters
n_input = sen_fix_len*word_len
n_out = 100 
drop_out = 0.9 


def get_sentence_vector(sentence):
    global model
    global sen_fix_len
    words = jieba.cut(sentence)
    sentence_vec = []
    cnt = 0
    for word in words:
        try:
            word_vec = model[word]
        except:  # 将词向量中不存在的词填充
            continue
        if cnt >= sen_fix_len:
            break
        # word_vec = [2] * 300
        sentence_vec.append(word_vec)
        cnt += 1

    if len(sentence_vec) < sen_fix_len:  
       sentence_vec.extend([[0.0]*300]*(sen_fix_len - len(sentence_vec)))

    #print np.array(sentence_vec).reshape(-1).shape
    return np.array(sentence_vec).reshape(-1)

def gen_entity_text_vec():
    entity_text_dict_vec = {}
    data_dir = "./data/entity_text/"
    filelist = os.listdir(data_dir) 
    for filename in filelist:
        filepath = data_dir + filename
        with open(filepath) as fr:
            text = fr.read()
            text_vec = get_sentence_vector(text)
            entity_text_dict_vec[filename] = text_vec
    return entity_text_dict_vec


def gen_mention_text_vec():
    mention_text_dict_vec = {} 
    data_dir = "./data/mention_text/"
    filelist = os.listdir(data_dir)
    for filename in filelist:
        # print filename
        filepath = data_dir + filename
        with open(filepath) as fr:
            text = fr.read()
            text_vec = get_sentence_vector(text)
            mention_text_dict_vec[filename] = text_vec
    #pickle.dump(mention_text_dict_vec, open("mention_text_dict_vec.pk","w"))
    return mention_text_dict_vec

def create_mention_entity_pairs(mention_text_dict_vec, entity_text_dict_vec):
    pairs = []
    labels = []
    with open("./data/eng_gold.tab") as fr:
        cnt = 0
        for line in fr:
            if cnt >= tr_size:
                break
            #print line
            tokens = line.split("\t")
            mention_text_id = tokens[3].split(":")[0]
            entity_text_id = tokens[4]
            m_type = tokens[6]
            if "NOM" in m_type:
                continue
            if "NIL" in entity_text_id:
                continue
            #print cnt
            cnt += 2 
            mention_text_vec = mention_text_dict_vec[mention_text_id+".xml"]
            entity_text_vec = entity_text_dict_vec[entity_text_id]
            pairs += [[mention_text_vec, entity_text_vec]]
            labels += [1]
            
	    # error pairs.
            entity_keys = entity_text_dict_vec.keys()
            entity_random_id = entity_keys[random.randint(0, len(entity_keys)-1)]
            while entity_random_id == entity_text_id:
                entity_random_id = entity_keys[random.randint(0, len(entity_keys) - 1)]
            entity_text_vec = entity_text_dict_vec[entity_random_id]
            pairs+= [[mention_text_vec, entity_text_vec]]
            labels += [0]

    return np.array(pairs), np.array(labels)


print "load word2vec model"
t_start = time.time()
model = gensim.models.Word2Vec.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True)
print "load word2vec model finsh, use time %.3fs" % (time.time() - t_start)

print "gen entity_text_dict"
t_start = time.time()
if os.path.isfile("entity_text_dict_vec.pk"):
    entity_text_dict_vec = pickle.load(open("entity_text_dict_vec.pk"))
else:
    entity_text_dict_vec = dict(gen_entity_text_vec())
print "gen entity_text_dict_vec ok, use time %.3fs" % (time.time() - t_start)

print "gen mention_text_dict"
t_start = time.time()
if os.path.isfile("mention_text_dict_vec.pk"):
    mention_text_dict_vec = pickle.load(open("mention_text_dict_vec.pk"))
else:
    mention_text_dict_vec = dict(gen_mention_text_vec())
print "gen mention_text_dict_vec ok, use time %.3fs" % (time.time() - t_start)

print "create_mention_entity_pairs"
t_start = time.time()
tr_pairs, tr_labels = create_mention_entity_pairs(mention_text_dict_vec, entity_text_dict_vec) 
print "create_mention_entity_pairs ok, use time%.3fs" % (time.time() - t_start)



