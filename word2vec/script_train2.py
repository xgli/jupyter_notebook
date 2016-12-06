#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, codecs
import gensim, logging, multiprocessing
reload(sys)
sys.setdefaultencoding('utf-8')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python script.py infile outfile"
        sys.exit()
    infile, outfile = sys.argv[1:3]
    model = gensim.models.Word2Vec(gensim.models.word2vec.LineSentence(infile), size=400, window=5, min_count=5, sg=1, workers=multiprocessing.cpu_count())
    model.save(outfile)
    model.save_word2vec_format(outfile + '.vector2', binary=False)
