#!/usr/bin/env python
import re
import os
import os.path
import logging
from time import time
import math
import random
import codecs
import sqlite3
import sys
import cPickle
import mmap

##############################################################################
# Parameters
#

datadir = "/mnt/external/NLP"

# Maximal prefix N-gram size
N_max_prefix_size = 4

# Number of suggestions
N_sug = 5

##############################################################################
# Logging
#

formatter = logging.Formatter('%(asctime)s.%(msecs).03d - %(name)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')
file_handler = logging.FileHandler("combined.log", mode="w", encoding="UTF8")
file_handler.setFormatter(formatter)
stderr_handler = logging.StreamHandler()
stderr_handler.setFormatter(formatter)

log_main = logging.getLogger("Main")
log_main.addHandler(file_handler)
log_main.addHandler(stderr_handler)
log_main.setLevel(logging.DEBUG)

log_IKN = logging.getLogger("IKN")
log_IKN.addHandler(file_handler)
log_IKN.addHandler(stderr_handler)
log_IKN.setLevel(logging.INFO)

log_Bayes = logging.getLogger("Bayes")
log_Bayes.addHandler(file_handler)
log_Bayes.addHandler(stderr_handler)
log_Bayes.setLevel(logging.INFO)


# Normalisation regexes for whole lines - may produce line breaks
line_rules = [
    # pre-compiled regex                                    replacement
    #
    # revert unicode punctuation to ascii
    [re.compile(u"[\u2018-\u201b\u2032\u2035`]"),           "'"],
    [re.compile(u"[\u201c-\u201f\u2033\u2034\u2036\u2037\u2039\u203a\u2057\xab\xbb]"), "\""],
    [re.compile(u"[\u2010-\u2015\u2043]"),                  "-"],
    [re.compile(u"[\u2024\u2027]"),                         "."],
    [re.compile(u"\u2025"),                                 ".."],
    [re.compile(u"\u2026"),                                 "..."],
    [re.compile(u"[\u2000-\u200d\u2060\u202f\u205f]+"),     " "],
    [re.compile(u"\u2063"),                                 ","],
    [re.compile(u"\u2052"),                                 "%"],
    [re.compile(u"[\u204e\u2055\u2062]"),                   "*"],
    [re.compile(u"\u2052"),                                 "%"],
    [re.compile(u"\u2064"),                                 "+"],
    # sub-sentence separators to comma or newline with trailing-leading spaces
    [re.compile("\\r"),                                     ""],
    [re.compile("^|$"),                                     " "],
    [re.compile("(,\\s*)+"),                                " , "],
    # coalesce adjacent subsentence delimiters
    [re.compile("(,\\s*)+"),                                " , "],
    # strip leading and trailing subsentence delimiters
    [re.compile("(,\\s*)+$"),                               ""],
    [re.compile("^(\\s*,)+"),                               ""],
    # finally: split at sentence delimiters
    [re.compile("[.!?]+"),                                  " \n "]
]

# Normalisation regexes for sub-sentences
sentence_rules = [
    # pre-compiled regex                                    replacement
    # (NOTE: must be here at front, others rely on its result)
    # more than one dashes are delimiter
    [re.compile("-{2,}"),                                   " "],
    # quotes, parentheses, underscores to space
    [re.compile("[()\":/_]"),                               " "],
    # zap non-alnums in front of words
    [re.compile(" [^a-zA-Z0-9,]+"),                         " "],
    # zap non-alnums at the end of words
    [re.compile("[^a-zA-Z0-9,]+ "),                         " "],
    # remove those without at least a letter
    [re.compile(" [^a-zA-Z,]+ "),                           " "],
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
    [re.compile(" i "),                                     " I "],
    [re.compile(" i'"),                                     " I'"],
    [re.compile(" u "),                                     " you "],
    # some abbreviations (NOTE: standalone only)
    [re.compile(" o' ", re.I),                              " of "],
    [re.compile(" ol' ", re.I),                             " old "],
    # not-in-word apostrophes to space
    [re.compile(" '|' "),                                   " "],
    # zap all single characters except 'I' and 'a'
    [re.compile(" [^IAa,] "),                               " "],
    # zap all words with invalid characters (valid: alnum, ', _, +, -)
    [re.compile(" [^ ]*[^ a-zA-Z0-9_'+-,][^ ]* "),          " _ "],
    # coalesce adjacent subsentence delimiters
    [re.compile("(,\\s*)+"),                                " , "],
    # coalesce whitespaces
    [re.compile("\\s+"),                                    " "]
]

last_progress_printout = time()
# Print progress messages only once in a second
def need_progress_printout():
    global last_progress_printout
    now = time()
    if (now - last_progress_printout) > 1:
        last_progress_printout = now
        return True
    return False

def split_to_sentences(line):
    '''Normalises a line and returns a list of its sentences (at least one)'''
    for rule in line_rules:
        line = rule[0].sub(rule[1], line.rstrip("\n"))

    result = []
    for sentence in line.split("\n"):
        for rule in sentence_rules:
            sentence = rule[0].sub(rule[1], sentence)
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    return result

def readPCK(filename):
    infile = open(filename, "r")
    #mm = mmap.mmap(infile.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
    #x = cPickle.loads(mm)
    x = cPickle.load(infile)
    #mm.close()
    infile.close()
    return x


log_main.info("Reading words;")
words = readPCK("{d}/words.pck".format(d=datadir))
total = sum(map(lambda w: w["occurences"], words))
N = len(words)
# cut the word list to the top-P-percent
wordset = dict(map(lambda w: (w["word"].lower(), w["word"]), words))
log_main.debug("Read words; n={n}".format(n=N))

# create a word dict for converting word to index
sdrow = dict((words[i]["word"], i) for i in range(N))

log_Bayes.debug("Reading Bayesian factors;")
m = readPCK("{d}/bayes_factors.pck".format(d=datadir))

log_IKN.debug("Opening databases;")
db = [ None for i in range(0, 1 + N_max_prefix_size) ]
qFollowers = [ None for i in range(0, 1 + N_max_prefix_size) ]
qPrefixes = [ None for i in range(0, 1 + N_max_prefix_size) ]
for i in range(1, 1 + N_max_prefix_size):
    filename = "{d}/all.{i}.db".format(d=datadir, i=i)
    db[i] = sqlite3.connect(filename)
    log_IKN.debug("Opened db; idx='{i}', file='{f}'".format(i=i, f=filename))
    q = db[i].cursor()
    q.execute("PRAGMA query_only = ON")
    q.execute("PRAGMA synchronous = OFF")
    q.execute("PRAGMA journal_mode = OFF")
    q.execute("PRAGMA secure_delete = OFF")
    q.execute("PRAGMA locking_mode = EXCLUSIVE")
    q.execute("PRAGMA mmap_size = 4294967296")
    q.close()
    qFollowers[i] = db[i].cursor()
    qPrefixes[i] = db[i].cursor()

def get_followers_ML(u):
    ''' Maximum-Likelihood algorithm '''
    n = len(u)
    if not n in range(1, 1 + N_max_prefix_size):
        return {}
    qFollowers[n].execute("SELECT follower, occurences FROM ngram_t WHERE prefix=:prefix", {"prefix": " ".join(u)})
    wset = dict(map(lambda x: (x[0], int(x[1])), qFollowers[n].fetchall()))
    cu = 0
    t = 0
    for k, cuw in wset.iteritems():
        t += 1
        cu += cuw

    p = {}
    for w, cuw in wset.iteritems():
        p[w] = float(cuw) / cu
    return p

def get_followers_IKN(u):
    ''' Interpolated Kneser-Ney algorithm '''
    n = len(u)
    if not n in range(1, 1 + N_max_prefix_size):
        log_IKN.debug("Returning empty IKN; len='{n}'".format(n=n))
        return {}
    qFollowers[n].execute("SELECT follower, occurences FROM ngram_t WHERE prefix=:prefix", {"prefix": " ".join(u)})
    wset = dict(map(lambda x: (x[0], int(x[1])), qFollowers[n].fetchall()))
    cu = 0
    t = 0
    for w, cuw in wset.iteritems():
        t += 1
        cu += cuw

    p = {}
    #d = dKN[n]
    d = int(float(cu) / t / 2) if t > 0 else 0
    if n <= 1:
        d = 0
    # first we 'borrow' at most d occurences from each word
    borrowed = 0
    for w, cuw in wset.iteritems():
        if cuw > d:
            borrowed += d
            p[w] = float(cuw - d) / cu
        else:
            borrowed += cuw
            p[w] = 0

    log_IKN.info("IKN('{u}'); n='{n}', t='{t}', cu='{cu}', d[n]='{d}', borrowed='{b}'".format(n=n, u=" ".join(u), t=t, cu=cu, d=d, b=borrowed))
    # then we redistribute them according to the (n-1)-gram probabilities
    if borrowed > 0:
        log_IKN.debug("Obtaining shorter-prefix stats; n='{n}'".format(n=n))
        prev_level = get_followers_IKN(u[1:])
        log_IKN.debug("Obtained shorter-prefix stats; n='{n}', len='{l}'".format(n=n, l=len(prev_level)))
        weights = {}
        total_pfx = 0
        for w, cuw in wset.iteritems():
            qPrefixes[n].execute("SELECT count(*) FROM ngram_t WHERE follower=:follower", {"follower": w})
            num_prefixes = float(qPrefixes[n].fetchone()[0])

            p_prev = prev_level[w] if w in prev_level else 0
            log_IKN.debug("Other prefixes; follower='{f}', p_prev='{p}', num_pfx='{n}'".format(f=w, p=p_prev, n=num_prefixes))
            weight = p_prev / num_prefixes

            weights[w] = weight
            total_pfx += weight

        borrowed = (float(borrowed) / cu / total_pfx) if cu > 0 and total_pfx > 0 else 0
        log_IKN.debug("Fixing ML probs; borrowed='{b}'".format(b=borrowed))
        for w, cuw in wset.iteritems():
            p[w] += borrowed * weights[w]
    return p


def get_combined_suggestions(sentence_words):
    row = [ 1 for i in range(N) ]
    for word in sentence_words:
        if not word in sdrow:
            continue
        j = sdrow[word]
        for i in range(N):
            row[i] *= m[j][i]

    for i in range(0, len(sentence_words)):
        prefix = sentence_words[i:]

        ngram_results = get_followers_ML(prefix)
        #ngram_results = get_followers_IKN(prefix)
        if not ngram_results:
            continue

        log_main.info("  Prefix: {p}".format(p=" ".join(prefix)))
        suggestions = []
        for k, v in ngram_results.iteritems():
            p = v
            if k in sdrow:
                odds = row[sdrow[k]]
                pbayes = odds / (1 + odds)
                log_main.debug("Combining result; word='{w}', p_ngram='{pn}', p_bayes='{pb}'".format(w=k, pn=v, pb=pbayes))
                p *= pbayes
            suggestions.append((p, k))

        suggestions.sort(reverse=True)
        #return suggestions[:N_sug]
        return suggestions
    return []


#tests = [
#          "^ the guy in front of me just bought a pound of bacon , a bouquet , and a case of",
#          "^ you're the reason why i smile everyday can you follow me please it would mean the",
#          "^ hey sunshine , can you follow me and make me the",
#          "^ very early observations on the bills game offense still struggling but the",
#          "^ go on a romantic date at the",
#          "^ well i'm pretty sure my granny has some old bagpipes in her garage i'll dust them off and be on my",
#          "^ is on tomorrow",
#          "^ love that film and haven't seen it in quite some",
#          "^ after the ice bucket challenge louis will push his long wet hair out of his eyes with his little",
#          "^ be grateful for the good times and keep the faith during the",
#          "^ if this isn't the cutest thing you've ever seen, then you must be"
#]
#
#for sentence in tests:
#    log_main.info("Sentence: '{t}'".format(t=sentence))
#    sentence_words = sentence.split(" ")
#    suggestions = get_combined_suggestions(sentence_words)
#    n = 0
#    for x in suggestions:
#        if x[0] <= 0:
#            break
#        log_main.info("    Hint: '{w}' ({p})".format(w=x[1], p=x[0]))
#        n += 1

tests = [
    { "sentence": "when you breathe , I want to be the air for you I'll be there for you , I'd live and I'd",
      "choices": ["give", "sleep", "die", "eat"] },
    { "sentence": "guy at my table's wife got up to go to the bathroom and I asked about dessert and he started telling me about his",
      "choices": ["spiritual", "financial", "marital", "horticultural"] },
    { "sentence": "I'd give anything to see arctic monkeys this",
      "choices": ["month", "weekend", "morning", "decade"] },
    { "sentence": "talking to your mom has the same effect as a hug and helps reduce your",
      "choices": ["stress", "sleepiness", "hunger", "happiness"] },
    { "sentence": "when you were in Holland you were like one inch away from me but you hadn't time to take a",
      "choices": ["walk", "picture", "minute", "look"] },
    { "sentence": "I'd just like all of these questions answered , a presentation of evidence , and a jury to settle the",
      "choices": ["incident", "account", "case", "matter"] },
    { "sentence": "I can't deal with unsymetrical things I can't even hold an uneven number of bags of groceries in each",
      "choices": ["finger", "hand", "toe", "arm"] },
    { "sentence": "every inch of you is perfect from the bottom to the",
      "choices": ["middle", "side", "center", "top"] },
    { "sentence": "I'm thankful my childhood was filled with imagination and bruises from playing",
      "choices": ["outside", "inside", "weekly", "daily"] },
    { "sentence": "I like how the same people are in almost all of Adam Sandler's",
      "choices": ["pictures", "stories", "movies", "novels"] }
]

ntest = 1
for test in tests:
    log_main.info("==== Sentence {n}: '{t}'".format(n=ntest, t=test["sentence"]))
    ntest += 1
    sentence_words = test["sentence"].split(" ")
    suggestions = get_combined_suggestions(sentence_words)
    result = dict((x[1], x[0]) for x in suggestions)
    found = 0
    for choice in test["choices"]:
        if choice in result:
            log_main.info("  Choice {c}: p={p}".format(c=choice, p=result[choice]))
            found += 1
        else:
            log_main.warning("  Choice {c} NOT FOUND".format(c=choice))
    n = 0
    for x in suggestions:
        if x[0] <= 0 or n > 10:
            break
        log_main.info("    Result: '{w}' ({p})".format(w=x[1], p=x[0]))
        n += 1


#def normalise_caps(w):
#    wl = w.lower()
#    return wordset[wl] if wl in wordset else "_"
#
#successfile = codecs.open("success.log", mode="w", encoding="utf-8")
#failfile = codecs.open("fail.log", mode="w", encoding="utf-8")
#infile = codecs.open("all.test.txt", mode="r", encoding="utf-8")
#random.seed(1519531190)
#total_lines = 0
#for sentence in infile:
#    # split and replace the rare words with '_'
#    common_words = map(lambda word: normalise_caps(word), sentence.rstrip("\n").split(" "))
#    if not common_words:
#        continue
#    # pull together multiple adjacent '_'-s (if any)
#    n = len(common_words) - 1
#    i = 0
#    while i < n:
#        if common_words[i] == "_":
#            while i < n and common_words[i + 1] == "_":
#                del common_words[i + 1]
#                n -= 1
#        i += 1
#
#    prefix_limit = 1 + int(random.random() * (len(common_words) - 2))
#    expected = common_words[prefix_limit]
#    if expected == "_" or expected == "^":
#        continue
#
#    suggestions = get_combined_suggestions(common_words[:prefix_limit])
#    s_str = " ".join("{w}({p})".format(w=x[1], p=x[0]) for x in suggestions)
#
#    success = False
#    for s in suggestions:
#        if s[1] == expected:
#            success = True
#            break
#
#    log_main.info("Check; success='{r}', expected='{e}', suggestions='{s}'".format(r=success, e=expected, s=s_str))
#    f = successfile if success else failfile
#    f.write("{n}:{e}: {s}\n".format(n=total_lines, e=expected, s=s_str))
#
#    total_lines += 1
#    if need_progress_printout():
#        log_main.debug("  Tested; lines='{l}'".format(l=total_lines))
#
#log_main.info("  Tested; lines='{l}'".format(l=total_lines))
#successfile.close()
#failfile.close()
#infile.close()

# vim: set et ts=4 sw=4:
