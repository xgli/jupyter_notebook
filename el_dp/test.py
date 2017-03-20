#!/usr/bin/env python
# coding=utf-8

mention_text_dir = "./data/eng_xml/"

with open("./data/eng_gold.tab") as fr:
    for line in fr:
        print line
        tokens = line.split("\t")
        mention_text_id = tokens[3].split(":")[0]
        start,end = tokens[3].split(":")[1].split('-')
        start = int(start)
        end = int(end)
        fr_mention = open(mention_text_dir+mention_text_id+".xml")
        t_lines = fr_mention.readlines()
        for t_line in t_lines:
            #print t_line
            loc = int(t_line.split("\t")[0])
            if loc <= start and loc+len(t_line) > start:
                print start
                print loc
                print loc + len(t_line)
                print len(t_line)
        
        #print text

