#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, codecs
import jieba.posseg as pseg
reload(sys)
sys.setdefaultencoding('utf-8')
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python script.py infile outfile"
        sys.exit()
    i = 0
    infile, outfile = sys.argv[1:3]
    output = codecs.open(outfile, 'w', 'utf-8')
    with codecs.open(infile, 'r', 'utf-8') as myfile:
        for line in myfile:
            line = line.strip()
            if len(line) < 1:
                continue
            if line.startswith('<doc'):
                i = i + 1
                if(i % 1000 == 0):
                    print('Finished ' + str(i) + ' articles')
                continue
            if line.startswith('</doc'):
                output.write('\n')
                continue
            words = pseg.cut(line)
            for word, flag in words:
                if flag.startswith('x'):
                    continue
                output.write(word + ' ')
    output.close()
    print('Finished ' + str(i) + ' articles')