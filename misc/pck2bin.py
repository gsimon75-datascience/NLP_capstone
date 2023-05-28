#!/usr/bin/python
import cPickle
import struct

filename = "final/en_US/all.normalised.sentences"
#filename = "sample.normalised.sentences"
infile = open(filename, mode="rb")
outfile = open("{f}.bin".format(f=filename), mode="wb")
try:
    while True:
        w = cPickle.load(infile)
        w.append(0xffffffff)
        nw = len(w)
        outfile.write(struct.pack("<{n}I".format(n=nw), *w))
except EOFError:
    pass

infile.close()
outfile.close()
# vim: set et ts=4 sw=4:
