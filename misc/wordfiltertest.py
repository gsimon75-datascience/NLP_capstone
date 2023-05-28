#!/usr/bin/env python
import re
import os
import os.path
import logging
from time import time, sleep
import random
import codecs
import sqlite3
import cPickle
import sys
import psutil
import itertools
import gc
import bisect
import math
from cStringIO import StringIO

# Normalisation regexes for whole lines - may produce line breaks
line_rules = [
    # pre-compiled regex                                    replacement
    #
    # revert unicode punctuation to ascii
    [re.compile(u"[\u2018-\u201b\u2032\u2035`]"),           "'"],
    [re.compile(u"[\u201c-\u201f\u2033\u2034\u2036\u2037\u2039\u203a\u2057\xab\xbb]"), "\""],
    [re.compile(u"[\u2010-\u2015\u2043\xad]"),              "-"],
    [re.compile(u"[\u2024\u2027]"),                         "."],
    [re.compile(u"\u2025"),                                 ".."],
    [re.compile(u"\u2026"),                                 "..."],
    [re.compile(u"[\u2000-\u200d\u2060\u202f\u205f]+"),     " "],
    [re.compile(u"\u2063"),                                 ","],
    [re.compile(u"\u2052"),                                 "%"],
    [re.compile(u"[\u204e\u2055\u2062]"),                   "*"],
    [re.compile(u"\u2052"),                                 "%"],
    [re.compile(u"\u2064"),                                 "+"],
    # no crlf, bom, lrm, rlm, etc.
    [re.compile(u"[\r\ufeff\u200e\u200f\x80-\xbf\xd7\xf7]"),""],
    # quotes, parentheses, underscores to space
    [re.compile("[][@{}<>()\"\\|~`:*/%_#$,;+=^0-9-]"),      " "],
    [re.compile("&\\S*"),                                   " "],
    [re.compile("^|$"),                                     " "],
]

# Normalisation regexes for sub-sentences
sentence_rules = [
    # pre-compiled regex                                    replacement
    # (NOTE: must be here at front, others rely on its result)
    # zap non-alpha in front of words
    [re.compile(" [^a-zA-Z]+"),                             " "],
    # zap non-alpha at the end of words
    [re.compile("[^a-zA-Z]+ "),                             " "],
    # remove those without at least a letter
    [re.compile(" [^a-zA-Z]+ "),                            " "],
    # some common typos (NOTE: standalone only)
    [re.compile(" aint ", re.I),                            " ain't "],
    [re.compile(" cant ", re.I),                            " can't "],
    [re.compile(" couldnt ", re.I),                         " couldn't "],
    [re.compile(" doesnt ", re.I),                          " doesn't "],
    [re.compile(" dont ", re.I),                            " don't "],
    [re.compile(" didnt ", re.I),                           " didn't "],
    [re.compile(" hasnt ", re.I),                           " hasn't "],
    [re.compile(" havent ", re.I),                          " haven't "],
    [re.compile(" hadnt ", re.I),                           " hadn't "],
    [re.compile(" isnt ", re.I),                            " isn't "],
    [re.compile(" arent ", re.I),                           " aren't "],
    [re.compile(" wasnt ", re.I),                           " wasn't "],
    [re.compile(" werent ", re.I),                          " weren't "],
    [re.compile(" wont ", re.I),                            " won't "],
    [re.compile(" wouldnt ", re.I),                         " wouldn't "],
    [re.compile(" mustnt ", re.I),                          " mustn't "],
    [re.compile(" neednt ", re.I),                          " needn't "],
    [re.compile(" oughtnt ", re.I),                         " oughtn't "],
    [re.compile(" shant ", re.I),                           " shan't "],
    [re.compile(" shouldnt ", re.I),                        " shouldn't "],
    [re.compile(" id ", re.I),                              " I'd "],
    [re.compile(" youd ", re.I),                            " you'd "],
    [re.compile(" hed ", re.I),                             " he'd "],
    [re.compile(" youd ", re.I),                            " you'd "],
    [re.compile(" theyd ", re.I),                           " they'd "],
    [re.compile(" ive ", re.I),                             " I've "],
    [re.compile(" youve ", re.I),                           " you've "],
    [re.compile(" weve ", re.I),                            " we've "],
    [re.compile(" theyve ", re.I),                          " they've "],
    [re.compile(" im ", re.I),                              " I'm "],
    [re.compile(" youre ", re.I),                           " you're "],
    [re.compile(" hes ", re.I),                             " he's "],
    [re.compile(" shes ", re.I),                            " she's "],
    [re.compile(" theyre ", re.I),                          " they're "],
    [re.compile(" ill ", re.I),                             " I'll "],
    [re.compile(" youll ", re.I),                           " you'll "],
    [re.compile(" itll ", re.I),                            " it'll "],
    [re.compile(" theyll ", re.I),                          " they'll "],
    [re.compile(" couldve ", re.I),                         " could've "],
    [re.compile(" shouldve ", re.I),                        " should've "],
    [re.compile(" wouldve ", re.I),                         " would've "],
    [re.compile(" lets ", re.I),                            " let's "],
    [re.compile(" thats ", re.I),                           " that's "],
    [re.compile(" heres ", re.I),                           " here's "],
    [re.compile(" theres ", re.I),                          " there's "],
    [re.compile(" whats ", re.I),                           " what's "],
    [re.compile(" whos ", re.I),                            " who's "],
    [re.compile(" wheres ", re.I),                          " where's "],
    [re.compile(" noones ", re.I),                          " noone's "],
    [re.compile(" everyones ", re.I),                       " everyone's "],
    [re.compile(" nowheres ", re.I),                        " nowhere's "],
    [re.compile(" everywheres ", re.I),                     " everywhere's "],
    [re.compile(" yall ", re.I),                            " y'all "],
    [re.compile(" bday ", re.I),                            " birthday "],
    [re.compile(" dis ", re.I),                             " this "],
    [re.compile(" dat ", re.I),                             " that "],
    [re.compile(" i "),                                     " I "],
    [re.compile(" i'"),                                     " I'"],
    # some abbreviations (NOTE: standalone only)
    [re.compile(" c'mon ", re.I),                           " come on "],
    [re.compile(" (be?)?c[ou]z ", re.I),                    " because "],
    [re.compile(" imma ", re.I),                            " I'm a "],
    [re.compile(" ofcoz ", re.I),                           " of course "],
    [re.compile(" pl[sz] ", re.I),                          " please "],
    [re.compile(" ppl ", re.I),                             " people "],
    [re.compile(" tho ", re.I),                             " though "],
    [re.compile(" u ", re.I),                               " you "],
    [re.compile(" ur ", re.I),                              " your "],
    [re.compile(" o' ", re.I),                              " of "],
    [re.compile(" ol' ", re.I),                             " old "],
    # zap all single characters except 'I' and 'a'
    [re.compile(" [^IAa] "),                               " "],
    # zap all words with invalid characters (valid: alnum, ', _)
    #[re.compile(" [^ ]*[^, a-zA-Z0-9_'][^ ]* "),          " _ "],
    # everything where a letter repeats more than 2 times
    [re.compile(" \S*(\S)\\1{3,}\S* ", re.I),               " "],
    # coalesce whitespaces
    [re.compile("\\s+"),                                    " "]
]


infile = codecs.open("words.100", mode="r", encoding="utf-8")
outfile = codecs.open("words.out", mode="w", encoding="utf-8")

for line in infile:
    for rule in line_rules:
        line = rule[0].sub(rule[1], line)

    result = []
    for sentence in line.split("\n"):
        print "input:  >" + sentence + "<"
        if not sentence.strip():
            continue
        for rule in sentence_rules:
            sentence = rule[0].sub(rule[1], sentence)
        sentence = sentence.strip()
        if sentence:
            print "output: >" + sentence + "<"
            outfile.write(sentence + "\n")

outfile.close()
infile.close()

### # Normalisation regexes for whole lines - may produce line breaks
### line_rules = [
###     # pre-compiled regex                                    replacement
###     #
###     # revert unicode punctuation to ascii
###     [re.compile(u"[\u2018-\u201b\u2032\u2035`]"),           "'"],
###     [re.compile(u"[\u201c-\u201f\u2033\u2034\u2036\u2037\u2039\u203a\u2057\xab\xbb]"), "\""],
###     [re.compile(u"[\u2010-\u2015\u2043\xad]"),              "-"],
###     [re.compile(u"[\u2024\u2027]"),                         "."],
###     [re.compile(u"\u2025"),                                 ".."],
###     [re.compile(u"\u2026"),                                 "..."],
###     [re.compile(u"[\u2000-\u200d\u2060\u202f\u205f]+"),     " "],
###     [re.compile(u"\u2063"),                                 ","],
###     [re.compile(u"\u2052"),                                 "%"],
###     [re.compile(u"[\u204e\u2055\u2062]"),                   "*"],
###     [re.compile(u"\u2052"),                                 "%"],
###     [re.compile(u"\u2064"),                                 "+"],
###     # no crlf, bom, lrm, rlm, etc.
###     [re.compile(u"[\r\ufeff\u200e\u200f\x80-\xbf\xd7\xf7]")," "],
###     # quotes, parentheses, underscores to space
###     [re.compile("[][{}<>()\"\\|~`:*/%_#$,;+=^0-9-]"),       " "],
###     [re.compile("&\\S*"),                                   " "],
###     [re.compile("^|$"),                                     " "],
### ]
### 
### # Normalisation regexes for sub-sentences
### sentence_rules = [
###     # pre-compiled regex                                    replacement
###     # (NOTE: must be here at front, others rely on its result)
###     # zap non-alpha in front of words
###     [re.compile(" [^a-zA-Z]+"),                             " "],
###     # zap non-alpha at the end of words
###     [re.compile("[^a-zA-Z]+ "),                             " "],
###     # remove those without at least a letter
###     [re.compile(" [^a-zA-Z]+ "),                            " "],
###     # not-in-word apostrophes to space
###     [re.compile(" '+"),                                     " "],
###     [re.compile("'+ "),                                     " "],
###     # zap all single characters except 'I' and 'a'
###     [re.compile(" [^IAa] "),                                " "],
###     # zap all words with invalid characters (valid: alnum, ', _)
###     #[re.compile(" [^ ]*[^, a-zA-Z0-9_'][^ ]* "),            " _ "],
###     # everything where a letter repeats more than 2 times
###     [re.compile(" \S*(\S)\\1{3,}\S* "),                     " "],
###     # coalesce whitespaces
###     [re.compile("\\s+"),                                    " "]
### ]
# vim: set et ts=4 sw=4:
