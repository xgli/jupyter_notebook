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

class ELCNN(object):
    """
    A CNN for entity link.
    Uses an convolutional, max-pooling and softmax layer
    """
    def __init__(self,sen_fix_len,word_len,tr_size,batch_size,epoch_size,learn_rate,display_step):
       
       #tf Graph Input
        self.images_L = tf.placeholder(tf.float32,shape=([None,n_input]),name='L')
        self.images_R = tf.placeholder(tf.float32,shape=([None,n_input]),name='R')
        self.labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
        self.dropout_f = tf.placeholder("float")




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


weight = {
    #the first convolutional layer w
    'w1':[5,5,1,10],
    #the seconed convolutional layer w
    'w2':[5,5,10,5],
    #densely  connected layer,
    'wd1':[(sen_fix_len/4)*(word_len/4)*5,(sen_fix_len/4)*(word_len/4)*5],
    #readout layer
    'out':[(sen_fix_len/4)*(word_len/4)*5,n_out]
}

bias = {
    #the first convolutional layer w
    'bc1':[weight['w1'][-1]],
    #the seconed convolutional layer w
    'bc2':[weight['w2'][-1]],
    #densely  connected layer,
    'bd1':[weight['wd1'][-1]],
    #readout layer
    'out':[n_out]
}

print weight
print bias

def mclnet(x,_dropout):
    # first convolutional layer
    x_image = tf.reshape(x, [-1, sen_fix_len, word_len, 1])
    W_conv1 = weight_variable(weight['w1'],"W_conv1")
    b_conv1 = bias_variable(bias['bc1'],"b_conv1")   
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # seconed convolutional layer
    W_conv2 = weight_variable(weight['w2'], "W_conv2")
    b_conv2 = bias_variable(bias['bc2'], "b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable(weight['wd1'],"W_fc1")
    b_fc1 = bias_variable(bias['bd1'],"b_fc1")

    h_pool2_flat = tf.reshape(h_pool2,[-1,weight['wd1'][0]])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # droput 
    h_fc1_drop = tf.nn.dropout(h_fc1, _dropout)

    # readout layer
    W_fc2 = weight_variable(weight['out'],"W_out")
    b_fc2 = bias_variable(bias['out'],"b_out")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    return y_conv

# weight initialization
def weight_variable(shape, name="W"):
    with tf.variable_scope(name):
        return tf.get_variable("W",shape)

def bias_variable(shape,name="b"):
    with tf.variable_scope(name):
        return tf.get_variable("b",shape)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def contrastive_loss(y,d):
    #tmp= y *tf.square(d)
    tmp= tf.mul(y,tf.square(d))
    tmp2 = tf.mul((1-y),tf.square(tf.maximum((1 - d),0)))
    #tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(tmp +tmp2)/batch_size/2

def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() < 0.5].mean()
 # return tf.reduce_mean(labels[prediction.ravel() < 0.5])

global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 0.001

learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,10,0.1,staircase=True)


with tf.variable_scope("siamese") as scope:
	model1 = mclnet(images_L, dropout_f)
	scope.reuse_variables()
	model2 = mclnet(images_R, dropout_f)


predict  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,predict)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

#output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs",timestamp))
print("writing to {}\n".format(out_dir))

#summaries for loss and accuracy
loss_summary = tf.scalar_summary("loss",loss)
#acc_summary = tf.scalar_summary("accuracy",)

#train summaries
train_summary_op = tf.merge_summary([loss_summary])
train_summary_dir = os.path.join(out_dir,"summaries","train")

#checkpointing
checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir,"model")
#tensorflow assumes this directory already exists so we need to create it
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.all_variables())


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



init = tf.initialize_all_variables()

with tf.Session() as sess:
	# sess.run(init)
	print "init variables"
	sess.run(init)
	train_summary_writer = tf.train.SummaryWriter(train_summary_dir,sess.graph)
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
			input1,input2,y =next_batch(s,e,tr_pairs,tr_labels)
                        print input1.shape
                        print input2.shape
                        print y.shape
			_,loss_value,predict,summaries =sess.run([train_op,loss,predict,train_summary_op],
				 feed_dict={images_L:input1,images_R:input2 ,labels:y,dropout_f:drop_out})
			train_summary_writer.add_summary(summaries,epoch*total_batch + i)
			current_step = tf.train.global_step(sess,global_step)
			path = saver.save(sess,checkpoint_prefix,global_step=current_step)
			tr_acc = compute_accuracy(predict,y)
			if math.isnan(tr_acc) and epoch != 0:
				print('tr_acc %0.2f' % tr_acc)
				continue
			avg_loss += loss_value
			avg_acc +=tr_acc*100
                        print "loss_value:%f" % loss_value
# print('epoch %d loss %0.2f' %(epoch,avg_loss/total_batch))
		duration = time.time() - start_time
		print('epoch %d  time: %f loss %0.5f acc %0.2f' 
                   %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))
                if epoch % 5 == 0:
                    acc = 0.
                    for i in range(total_batch):
                        s = i* batch_size
                        e = (i+1)*batch_size
                        input1,input2,y = next_batch(s,e,tr_pairs,tr_labels)
                        prediction = distance.eval(feed_dict={images_L:input1,images_R:input2, labels:y,dropout_f:1.0})
                        acc += compute_accuracy(prediction,y)*100
                    print("accuract training set %0.2f" %  (acc/total_batch))
                        


