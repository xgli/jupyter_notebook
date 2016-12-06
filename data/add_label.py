#!/usr/bin/env python
# coding=utf-8
filename = "./pigai.txt"

fw = open("pigai_lable.txt","w")
with open(filename) as fr:
    for line in fr:
        line = line.rstrip()
        outline = line + "\t"  + "__label__normal\n"
        fw.write(outline)
fw.close()
