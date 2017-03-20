# coding: utf-8
#初步跑通程序

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

# load word2vec
m_sen_fix_len = 80 #句子长度 
e_sen_fix_len = 20 #句子长度 
tr_size = 4000*2 #
batch_size = 128 
epoch_size = 40

def get_m_sentence_vector(sentence):
    global model
    global m_sen_fix_len
    words = jieba.cut(sentence)
    sentence_vec = []
    cnt = 0
    for word in words:
        try:
            word_vec = model[word]
        except:  # 将词向量中不存在的词填充
            continue
        if cnt >= m_sen_fix_len:
            break
        # word_vec = [2] * 300
        sentence_vec.append(word_vec)
        cnt += 1

    if len(sentence_vec) < m_sen_fix_len:  
       sentence_vec.extend([[0.0]*300]*(m_sen_fix_len - len(sentence_vec)))

    #print np.array(sentence_vec).reshape(-1).shape
    #print len(sentence_vec)

    #print "m"
    #print np.array(sentence_vec).reshape(-1).shape
    return np.array(sentence_vec).reshape(-1)


def get_e_sentence_vector(sentence):
    global model
    global e_sen_fix_len
    words = jieba.cut(sentence)
    sentence_vec = []
    cnt = 0
    for word in words:
        try:
            word_vec = model[word]
        except:  # 将词向量中不存在的词填充
            continue
        if cnt >= e_sen_fix_len:
            break
        # word_vec = [2] * 300
        sentence_vec.append(word_vec)
        cnt += 1

    if len(sentence_vec) < e_sen_fix_len:  
       sentence_vec.extend([[0.0]*300]*(e_sen_fix_len - len(sentence_vec)))

    #print np.array(sentence_vec).reshape(-1).shape
    #print len(sentence_vec)
    #print "e"
    #print np.array(sentence_vec).reshape(-1).shape
    return np.array(sentence_vec).reshape(-1)




def gen_entity_text_vec():
    entity_text_dict_vec = {}
    data_dir = "./data/entity_text/"
    filelist = os.listdir(data_dir) 
    for filename in filelist:
        #print filename
        filepath = data_dir + filename
        with open(filepath) as fr:
            text = fr.read()
            text_vec = get_e_sentence_vector(text)
            entity_text_dict_vec[filename] = text_vec
            
    #pickle.dump(entity_text_dict_vec,open("entity_text_dict_vec.pk","w"))     
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
            text_vec = get_m_sentence_vector(text)
            mention_text_dict_vec[filename] = text_vec
    #pickle.dump(mention_text_dict_vec, open("mention_text_dict_vec.pk","w"))
    return mention_text_dict_vec

def create_mention_entity_pairs(mention_text_dict_vec, entity_text_dict_vec):
    #pairs = []
    inputs1 = []
    inputs2 = []
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
            #mention_text_vec = get_sentence_vector(mention_text)
            entity_text_vec = entity_text_dict_vec[entity_text_id]
            #entity_text_vec = get_sentence_vector(entity_text)
            inputs1 += [mention_text_vec]
            inputs2 += [entity_text_vec]
    #        pairs += [[mention_text_vec, entity_text_vec]]
            labels += [1]

            # error pairs.
            entity_keys = entity_text_dict_vec.keys()
            entity_random_id = entity_keys[random.randint(0, len(entity_keys)-1)]
            while entity_random_id == entity_text_id:
                entity_random_id = entity_keys[random.randint(0, len(entity_keys) - 1)]
            entity_text_vec = entity_text_dict_vec[entity_random_id]
            inputs1 += [mention_text_vec]
            inputs2 += [entity_text_vec]
    #        pairs+= [[mention_text_vec, entity_text_vec]]
            labels += [0]

    #print np.array(pairs)[0][0].shape
    #print np.array(pairs)[0][1].shape

    return np.array(inputs1),np.array(inputs2),np.array(labels)



# weight initialization
def weight_variable(shape, name="W"):
	initial = tf.truncated_normal(shape=shape,stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape,name="b"):
	initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def mention_net(x,_dropout):
    # first convolutional layer
    x_image = tf.reshape(x, [-1,m_sen_fix_len,300,1])
    #x_image = tf.cast(x_image,tf.float32)
    W_conv1 = weight_variable([5,5,1,10],"W_conv1")
    b_conv1 = bias_variable([10],"b_conv1")   
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # seconed convolutional layer
    W_conv2 = weight_variable([5,5,10,5], "W_conv2")
    b_conv2 = bias_variable([5], "b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # densely connected layer
    W_fc1 = weight_variable([(m_sen_fix_len/4)*75*5, (m_sen_fix_len/4)*75*5],"W_fc1")
    b_fc1 = bias_variable([(m_sen_fix_len/4)*75*5],"b_fc1")

    h_pool2_flat = tf.reshape(h_pool2,[-1,(m_sen_fix_len/4)*75*5])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # droput 
    h_fc1_drop = tf.nn.dropout(h_fc1, _dropout)
    # h_fc1_drop = tf.nn.dropout(h_fc1, )

    # readout layer
    W_fc2 = weight_variable([(m_sen_fix_len/4)*75*5,100],"W_fc2")
    b_fc2 = bias_variable([100],"b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    return y_conv


def entity_net(x,_dropout):
    # first convolutional layer
    x_image = tf.reshape(x, [-1,e_sen_fix_len,300,1])
    #x_image = tf.cast(x_image,tf.float32)
    W_conv1 = weight_variable([5,5,1,10],"W_conv1")
    b_conv1 = bias_variable([10],"b_conv1")   
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # seconed convolutional layer
    W_conv2 = weight_variable([5,5,10,5], "W_conv2")
    b_conv2 = bias_variable([5], "b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # densely connected layer
    W_fc1 = weight_variable([(e_sen_fix_len/4)*75*5, (e_sen_fix_len/4)*75*5],"W_fc1")
    b_fc1 = bias_variable([(e_sen_fix_len/4)*75*5],"b_fc1")

    h_pool2_flat = tf.reshape(h_pool2,[-1,(e_sen_fix_len/4)*75*5])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # droput 
    h_fc1_drop = tf.nn.dropout(h_fc1, _dropout)
    # h_fc1_drop = tf.nn.dropout(h_fc1, )

    # readout layer
    W_fc2 = weight_variable([(e_sen_fix_len/4)*75*5,100],"W_fc2")
    b_fc2 = bias_variable([100],"b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    return y_conv




def contrastive_loss(y,d):
    #tmp= y *tf.square(d)
    tmp= tf.mul(y,tf.square(d))
    tmp2 = tf.mul((1-y),tf.square(tf.maximum((1 - d),0)))
    #tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2

def compute_accuracy(prediction,labels):
    #return labels[prediction.ravel() < 0.5].mean()
    a = labels[prediction.ravel() < 0.5]
    #print a
    return a.mean()

# return tf.reduce_mean(labels[prediction.ravel() < 0.5])
def next_batch(s,e,inputs,labels):
    input1 = inputs[s:e,0]
    input2 = inputs[s:e,1]
    print input1.shape
    print input2.shape
    print input1[0].shape
    print input2[0].shape
    y = np.reshape(labels[s:e],(batch_size,1))
    input1 = input1.reshape(-1)
    print input1.shape
    print input1
    #.reshape(-1,m_sen_fix_len*300)
    input2 = input2.reshape(-1).reshape(-1,e_sen_fix_len*300)
    #input1 = np.reshape(input1,(batch_size,m_sen_fix_len*300))  
    #input2 = np.reshape(input2,(batch_size,e_sen_fix_len*300))  
    #y = np.reshape(labels[s:e],(batch_size,1))
    return np.array(input1),np.array(input2),np.array(y)

def next(s,e,inputs1,inputs2,labels):
    input1 = inputs1[s:e]
    input2 = inputs2[s:e]
    y = np.reshape(labels[s:e],(batch_size,1))
    #print input1.shape
    #print input2.shape
    return np.array(input1),np.array(input2),y


global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)

images_L = tf.placeholder(tf.float32,shape=([None,m_sen_fix_len*300]),name='L')
images_R = tf.placeholder(tf.float32,shape=([None,e_sen_fix_len*300]),name='R')
labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
dropout_f = tf.placeholder("float")


with tf.variable_scope("siamese") as scope:
	model1 = mention_net(images_L, dropout_f)
	model2 = entity_net(images_R, dropout_f)


distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance)

batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(loss)

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

del model

print "create_mention_entity_pairs"
t_start = time.time()
inputs1, inputs2,tr_labels = create_mention_entity_pairs(mention_text_dict_vec, entity_text_dict_vec) 
print "create_mention_entity_pairs ok, use time%.3fs" % (time.time() - t_start)

del entity_text_dict_vec
del mention_text_dict_vec



with tf.Session() as sess:
	# sess.run(init)
	print "init variables"
	tf.initialize_all_variables().run()
	print "init variables ok"
	# Training cycle
	for epoch in range(epoch_size):
        	avg_loss = 0.
		avg_acc = 0.
        	total_batch = tr_size / batch_size - 1 
		start_time = time.time()
		# Loop over all batches
		for i in range(total_batch):
			s  = i * batch_size
			e = (i+1) *batch_size
			input1,input2,y =next(s,e,inputs1,inputs2,tr_labels)
			_,loss_value,predict=sess.run([optimizer,loss,distance], feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:0.9})
			tr_acc = compute_accuracy(predict,y)
			if math.isnan(tr_acc) and epoch != 0:
				print('tr_acc %0.2f' % tr_acc)
				continue
			avg_loss += loss_value
			avg_acc +=tr_acc*100
		duration = time.time() - start_time
		print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
                if epoch % 5 == 0:
                    acc = 0.
                    for i in range(total_batch):
                        s = i* batch_size
                        e = (i+1)*batch_size
                        input1,input2,y = next(s,e,inputs1,inputs2,tr_labels)
                        prediction = distance.eval(feed_dict={images_L:input1,images_R:input2, labels:y,dropout_f:1.0})
                        acc += compute_accuracy(prediction,y)*100
                    print("accuract training set %0.2f" %  (acc/total_batch))
                        

