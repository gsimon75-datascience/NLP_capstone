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
import struct
from cStringIO import StringIO

##############################################################################
# Parameters
#

# Number of suggestions
N_sug = 3

# Maximal size of the suggestion dictionary
N_max = 2000000

# Maximal prefix N-gram size
N_max_prefix_size = 4

# 90% training, 10% testing
train_ratio = 1.0

##############################################################################
# Logging
#

log = logging.getLogger("Main")
#formatter = logging.Formatter('%(asctime)s.%(msecs).03d - %(name)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')
#formatter = logging.Formatter('%(levelname)8s - %(message)s', datefmt='%H:%M:%S')

file_handler = logging.FileHandler("build3.log", mode="w", encoding="UTF8")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stderr_handler = logging.StreamHandler()
stderr_handler.setFormatter(formatter)
log.addHandler(stderr_handler)

if "NLP_DEBUG" in os.environ:
    log.setLevel(int(os.environ["NLP_DEBUG"]))
else:
    log.setLevel(logging.DEBUG)


##############################################################################
# Parts of the training process
#

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
    [re.compile(" \S*(\S)\\1{3,}\S* ",re.I),                " "],
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

progress_counter = 0
progress_delta = 5
def sql_progress():
    global progress_counter, progress_delta
    progress_counter += progress_delta
    log.debug("SQL progress; steps='{n}M'".format(n=progress_counter))

def free_mb():
    return psutil.virtual_memory().available >> 20

# retain some memory, about 1 GB
def near_oom():
    return free_mb() <= 1024


def dump_memstat():
    log.debug("psutil.virtual_memory() = {x}".format(x=psutil.virtual_memory()))


def open_dict_db(db_filename):
    ''' Open the SQLite database for the dictionary, creating and initialising it if it doesn't yet exist'''
    log.info("Using dict db; db_filename='{f}'".format(f=db_filename))
    already_exists = os.path.isfile(db_filename)
    db = sqlite3.connect(db_filename)  # , isolation_level=None)
    global progress_delta
    #db.set_progress_handler(sql_progress, progress_delta * 1000000)
    q = db.cursor()
    q.execute("PRAGMA synchronous = OFF")
    #q.execute("PRAGMA journal_mode = OFF")
    q.execute("PRAGMA secure_delete = OFF")
    q.execute("PRAGMA locking_mode = EXCLUSIVE")
    q.execute("PRAGMA mmap_size = 4294967296")
    q.execute("PRAGMA threads = 2")
    if not already_exists:
        # freshly created, create tables and indices
        q.execute('''
        CREATE TABLE word_t (
            id         INTEGER NOT NULL PRIMARY KEY,
            word       TEXT NOT NULL,
            occurences INTEGER NOT NULL DEFAULT 0)''')
        q.execute("CREATE INDEX word_word_i ON word_t (word)")

        q.execute('''
        CREATE TABLE bayes_t (
            condition   INTEGER NOT NULL,
            conditional INTEGER NOT NULL,
            occurences  INTEGER NOT NULL DEFAULT 0,
            factor      REAL,
            PRIMARY KEY (condition, conditional))''')
        # condition, conditional REFERENCES word_t

        q.execute('''
        CREATE TABLE prefix_t (
            id          INTEGER NOT NULL,
            parent      INTEGER NOT NULL,
            word        INTEGER NOT NULL,
            occurences  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (parent, word))''')
        q.execute("CREATE INDEX prefix_id_i ON prefix_t (id)")

        q.execute('''
        CREATE TABLE ngram_t (
            prefix      INTEGER NOT NULL,
            follower    INTEGER NOT NULL,
            occurences  INTEGER NOT NULL DEFAULT 0,
            factor      REAL,
            PRIMARY KEY (prefix, follower))''')

    q.close()
    db.commit()
    return (db, already_exists)

class CounterUpdater:
    ''' Cached wrapper to increase DB-backed counters '''
    def __init__(self, cache_max_size, db, tablename, count_colname, *key_colnames):
        self.cache_max_size = cache_max_size
        self.cache_size = 0
        self.cache = {}
        self.progress = 0
        self.db = db
        self.num_keys = len(key_colnames)
        self.qIncrEntry = db.cursor()
        self.qNewEntry = db.cursor()
        self.sqlIncrEntry = "UPDATE {table} SET {occurences} = {occurences} + ?{n} WHERE {condition}".format(
                    table=tablename,
                    occurences=count_colname,
                    n=1+self.num_keys,
                    condition=" AND ".join([ "{k}=?{n}".format(k=key_colnames[i], n=1+i) for i in xrange(self.num_keys) ]))
        self.sqlNewEntry = "INSERT INTO {table} ({keys}, {occurences}) VALUES ({placeholders}, ?{n})".format(
                    table=tablename,
                    keys=", ".join(key_colnames),
                    occurences=count_colname,
                    placeholders=", ".join([ "?{n}".format(n=1+i) for i in xrange(self.num_keys) ]),
                    n=1+self.num_keys)

    def execute(self, params):
        ''' params[:-1]=the key values, params[-1]=the occurences to add '''
        self.qIncrEntry.execute(self.sqlIncrEntry, params)
        if self.qIncrEntry.rowcount <= 0:
            self.qNewEntry.execute(self.sqlNewEntry, params)
        self.progress += 1
        if need_progress_printout():
            log.debug("  Committed; keys='{t}', cache_size='{n}'".format(t=self.progress, n=self.cache_size))

    def commit_cache(self, level=None, cachelevel=None, params=[]):
        if level is None:
            level = self.num_keys
            log.debug("Committing cache;")
            dump_memstat()
            self.progress = 0
            cachelevel = self.cache

        params.append(None)
        if level == 1:
            params.append(None)
            for k, v in cachelevel.iteritems():
                params[-2] = k
                params[-1] = v
                self.execute(params)
                self.cache_size -= 1
                cachelevel[k] = None
            del params[-1]
        else:
            for k, v in cachelevel.iteritems():
                params[-1] = k
                self.commit_cache(level-1, v, params)
                cachelevel[k] = None
        del params[-1]
        cachelevel.clear()

        if level == self.num_keys:
            log.debug("  Committed; keys='{t}', cache_size='{n}'".format(t=self.progress, n=self.cache_size))
            dump_memstat()
            self.cache.clear()
            self.cache_size = 0
            gc.collect()
            self.db.commit()
            log.debug("  Flushed cache;")
            dump_memstat()

    def increment(self, by, *args):
        x = self.cache
        for i in args[:-1]:
            if not i in x:
                x[i] = {}
            x = x[i]

        if args[-1] in x:
            x[args[-1]] += by
        else:
            x[args[-1]] = by
            self.cache_size += 1
            #if self.cache_size >= self.cache_max_size:
            if near_oom():
                log.debug("Near OOM;")
                dump_memstat()
                self.commit_cache()
                gc.collect()
                self.db.commit()
                while near_oom():
                    log.error("Still near OOM; free_mb='{f}'".format(f=free_mb()))
                    dump_memstat()
                    gc.collect()
                    self.db.commit()
                    sleep(1)

    def close(self):
        self.commit_cache()
        self.qIncrEntry.close()
        self.qNewEntry.close()
        gc.collect()
        self.db.commit()


def split_train_test(input_filename, train_filename, test_filename, train_ratio):
    ''' Split an input file to train- and test-sets '''
    log.info("Separating training and testing data; input='{i}', train_ratio='{r}'".format(i=input_filename, r=train_ratio))
    if os.path.isfile(train_filename) and os.path.isfile(test_filename):
        return
    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    trainfile = codecs.open(train_filename, mode="w", encoding="utf-8")
    testfile = codecs.open(test_filename, mode="w", encoding="utf-8")
    total_lines = 0
    random.seed(1519531190)
    for sentence in infile:
        if random.random() < train_ratio:
            trainfile.write(sentence)
        else:
            testfile.write(sentence)

        total_lines += 1
        if need_progress_printout():
            log.debug("  Splitted; lines='{l}'".format(l=total_lines))

    log.info("  Splitted; lines='{l}'".format(l=total_lines))
    testfile.close()
    trainfile.close()
    infile.close()


def split_to_sentences(line):
    '''Normalises a line and returns a list of its sentences (at least one)'''
    for rule in line_rules:
        line = rule[0].sub(rule[1], line)

    result = []
    for sentence in line.split("\n"):
        if not sentence.strip():
            continue
        for rule in sentence_rules:
            sentence = rule[0].sub(rule[1], sentence)
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    return result

def get_wordset(db):
    ''' Get the global word statistics '''
    qGlobalWordList = db.cursor()
    qGlobalWordList.execute("SELECT id, word FROM word_t")
    words = map(lambda rec: (int(rec[0]), rec[1]), qGlobalWordList.fetchall())
    qGlobalWordList.close()
    return words


def normalise_sentences(db, input_filename, output_filename, collect_words=True):
    '''Normalise a raw corpus and split into sub-sentences'''

    if os.path.isfile(output_filename):
        log.info("Using normalised sentences; output='{o}'".format(o=output_filename))
        return

    temp_filename = "{of}.temp".format(of=output_filename)
    log.info("Collecting sentences; input='{i}', output='{o}'".format(i=input_filename, o=temp_filename))
    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(temp_filename, mode="w", encoding="utf-8")
    total_lines = 0
    total_words = 0
    wordset = {}
    for line in infile:
        for sentence in split_to_sentences(line):
            outfile.write(sentence + "\n")
            if not collect_words:
                continue
            for word in sentence.split(" "):
                if word in wordset:
                    wordset[word] += 1
                else:
                    wordset[word] = 1
                    total_words += 1
        total_lines += 1
        if need_progress_printout():
            log.debug("  Collected; lines='{l}', words='{w}'".format(l=total_lines, w=total_words))

    log.info("  Collected; lines='{l}', words='{w}'".format(l=total_lines, w=total_words))
    outfile.close()
    infile.close()

    words = [] if collect_words else get_wordset(db)
    if collect_words:
        # coalesce the words that differ only in capitalisation: less caps letters wins
        wordset = map(lambda w: [w[0].lower(), w[0], w[1]], wordset.iteritems())
        wordset.sort()
        i = 0
        n = len(wordset) - 1
        while i < n:
            if wordset[i][0] == wordset[i + 1][0]:
                wordset[i + 1][2] += wordset[i][2]
                del wordset[i]
                n -= 1
            else:
                i += 1
        log.info("Removed single words; remained='{n}'".format(n=n))

        # assign an index to those words that occur more than once
        indexed = set()
        i = 0
        for w in wordset:
            if (w[2] > 1) and not w[1] in indexed:
                indexed.add(w[1])
                words.append((i, w[1], w[2]))
                i += 1

        # store the global words in the dict database
        q = db.cursor()
        q.executemany("INSERT INTO word_t (id, word, occurences) VALUES (?1, ?2, ?3)", words)
        q.close()
        db.commit()

    log.info("Normalising sentences; input='{i}', output='{o}'".format(i=temp_filename, o=output_filename))
    words = dict(map(lambda w: (w[1].lower(), w[0]), words))

    infile = codecs.open(temp_filename, mode="r", encoding="utf-8")
    outfile = open(output_filename, mode="wb")
    binfile = open("{f}.bin".format(f=output_filename), mode="wb")
    total_lines = 0
    for sentence in infile:
        # split and replace the rare words with '_'
        sentence_words = map(lambda w: words[w], filter(lambda w: w in words, sentence.rstrip("\n").lower().split(" ")))
        if sentence_words:
            cPickle.dump(sentence_words, outfile, -1)
        sentence_words.append(0xffffffff)
        binfile.write(struct.pack("<{n}I".format(n=len(sentence_words)), *sentence_words))

        total_lines += 1
        if need_progress_printout():
            log.debug("  Normalised; lines='{l}'".format(l=total_lines))

    log.info("  Normalised; lines='{l}'".format(l=total_lines))
    binfile.close()
    outfile.close()
    infile.close()
    os.unlink(temp_filename)


def collect_bayes_factors(db, input_filename):
    ''' Collect the Bayes-factor matrix '''
    log.info("Collecting Bayes factors; infile='{i}'".format(i=input_filename))
    # iterate through the sentences and count the words that occur in the same sentence
    qBayesCounter = CounterUpdater(0x4000000, db, "bayes_t", "occurences", "condition", "conditional")
    infile = open(input_filename, "rb")
    total = 0
    total_lines = 0
    try:
        while True:
            w = list(set(cPickle.load(infile)))
            nw = len(w)
            for i in xrange(nw - 1):
                iidx = w[i]
                for j in xrange(i + 1, nw):
                    jidx = w[j]
                    qBayesCounter.increment(1, iidx, jidx)
                    qBayesCounter.increment(1, jidx, iidx)
                    total += 2

            total_lines += 1
            if need_progress_printout():
                log.debug("  Coincidence; total='{t}', lines='{l}'".format(t=total, l=total_lines))
    except EOFError:
        pass
    log.info("  Coincidence; total='{t}', lines='{l}'".format(t=total, l=total_lines))
    infile.close()
    qBayesCounter.close()

    q = db.cursor()
    log.info("Removing single Bayes pairs;")
    q.execute("DELETE FROM bayes_t WHERE occurences < 2")

    log.info("Calculating Bayes factors;")
    q.execute("UPDATE bayes_t SET factor = (1.0 * ?1 / (SELECT occurences FROM word_t WHERE id = bayes_t.conditional) - 1) / (1.0 * (SELECT occurences FROM word_t WHERE id=bayes_t.condition) / occurences - 1)", (total, ))
    log.debug("Calculated Bayes factors; rows='{r}'".format(r=q.rowcount))

    q.close()
    db.commit()


def collect_ngrams(db, input_filename):
    ''' Collect the n-grams from the given corpus '''
    log.info("Collecting ngrams; infile='{i}'".format(i=input_filename))
    infile = open(input_filename, "rb")

    # ask the db to free up RAM
    db.execute("PRAGMA shrink_memory")

    # NOTE: In fact we are collecting the (n+1)-grams, which then will be
    # treated as a word following an n-gram, but that splitting will happen
    # only later, now they are just (n+1)-grams.

    # prefixes = [id, count, [child-words], [child-nodes]]
    prefixes = [-1, -1, [], []]
    last_prefix_id = 0

    # ngrams is not a tree, but a list - one list for each column
    ngram_prefix = []
    ngram_follower = []
    ngram_occurences = []

    total_lines = 0
    total_ngrams = 0

    def dump(root, indent="  "):
        if root[0] == -1:
            log.debug("{i}word=-1, count=0".format(i=indent))

        for i in xrange(len(root[2])):
            ch = root[3][i]
            log.debug("{i}  word={w}, count={c}".format(i=indent, w=root[2][i], c=ch[1]))
            dump(ch, indent + "  ")

    try:
        while True:
            w = cPickle.load(infile)
            nw = len(w)
            #log.debug("Sequence; nw={nw}, w={w}".format(nw=nw, w=w))
            w.reverse()
            for start in xrange(0, nw - 1):
                nmax = min(N_max_prefix_size, nw - start)
                #log.debug("Loop of start; start={s}, nmax={n}".format(s=start, n=nmax))
                root = prefixes
                for i in xrange(start, start + nmax - 1):
                    word = w[i]
                    idx = bisect.bisect_left(root[2], word)
                    #log.debug("Adding {x} to: (idx={i}, n={n})".format(x=word, i=idx, n=len(root[2])))
                    #dump(prefixes)
                    if idx != len(root[2]) and root[2][idx] == word:
                        root = root[3][idx]
                        root[1] += 1
                    else:
                        last_prefix_id += 1
                        newroot = [last_prefix_id, 1, [], []]
                        root[2].insert(idx, word)
                        root[3].insert(idx, newroot)
                        root = newroot

                    prefix_id = root[0]
                    follower = w[i + 1]
                    idx1 = bisect.bisect_left(ngram_prefix, prefix_id)
                    if idx1 == len(ngram_prefix) or ngram_prefix[idx1] != prefix_id:
                        ngram_prefix.insert(idx1, prefix_id)
                        ngram_follower.insert(idx1, follower)
                        ngram_occurences.insert(idx1, 1)
                    else:
                        idx2 = bisect.bisect_right(ngram_prefix, prefix_id, idx1)
                        idx = bisect.bisect_left(ngram_follower, follower, idx1, idx2)
                        if idx == idx2 or ngram_follower[idx] != follower:
#                            ngram_prefix.insert(idx, prefix_id)
#                            ngram_follower.insert(idx, follower)
#                            ngram_occurences.insert(idx, 1)
#                        else:
#                            ngram_occurences[idx] += 1
                            pass

                    #log.debug("Result:")
                    #dump(prefixes)

                total_ngrams += nmax + 1

            total_lines += 1

            if need_progress_printout():
                log.debug("  N-gram gen; lines='{l}', prefixes='{n}', mem='{m}'".format(l=total_lines, n=total_ngrams, m=free_mb()))
    except EOFError:
        pass
    infile.close()

    return

    log.warning("Committing N-grams to DB;")
    #dump(prefixes)
        
    qCommitPrefixes = db.cursor()
    total_lines = [0]
    def commit_children_of(parent):
        for i in xrange(0, len(parent[2])):
            chword = parent[2][i]
            ch = parent[3][i]
            if ch[1] <= 1:  # don't store prefixes that occur only once
                continue;
            qCommitPrefixes.execute("INSERT INTO prefix_t (id, parent, word, occurences) VALUES (?1, ?2, ?3, ?4)", (ch[0], parent[0], chword, ch[1]))
            total_lines[0] += 1
            if need_progress_printout():
                log.debug("  Prefix commit; done='{t}', total='{n}'".format(t=total_lines[0], n=last_prefix_id))
            commit_children_of(ch)
        del parent[:]

    log.info("Committing prefixes; n='{n}'".format(n=last_prefix_id))
    commit_children_of(prefixes)
    qCommitPrefixes.close()

    q = db.cursor()
    total_lines = 0
    n = len(ngram_prefix)
    for i in xrange(0, n):
        if ngram_occurences[i] <= 1:    # don't store ngrams that occur only once
            continue
        q.execute("INSERT INTO ngram_t (prefix, follower, occurences) VALUES (?1, ?2, ?3)", (ngram_prefix[i], ngram_follower[i], ngram_occurences[i]))
        total_lines += 1
        if need_progress_printout():
            log.debug("  N-gram commit; done='{t}', total='{n}'".format(t=total_lines, n=n))
    q.close()

    db.commit()

def calculate_ngram_factors(db):
    ''' Calculate Bayes factors for ngrams '''
    q = db.cursor()
    q.execute("SELECT SUM(occurences) FROM word_t")
    total = q.fetchone()[0]
    log.debug("Total words; n={n}".format(n=total))

    log.info("Calculating N-gram factors;")
    q.execute("UPDATE ngram_t SET factor = (1.0 * ?1 / (SELECT occurences FROM word_t WHERE id=ngram_t.follower) - 1) / (1.0 * (SELECT occurences FROM prefix_t WHERE id = ngram_t.prefix) / occurences - 1)", (total, ))
    log.debug("Calculated N-gram factors; rows='{r}'".format(r=q.rowcount))

    #q.execute("DROP INDEX prefix_id_i")

    db.commit()


##############################################################################
# The training process
#

def train_input(dict_db_filename, basename):
    ''' The whole training process from original input to dictionary db '''
    corpus_filename = "{b}.txt".format(b=basename)
    train_filename = "{b}.train.txt".format(b=basename)
    test_filename = "{b}.test.txt".format(b=basename)
    normalised_sentences_filename = "{b}.normalised.sentences".format(b=basename)

    (db, already_exists) = open_dict_db(dict_db_filename)
    split_train_test(corpus_filename, train_filename, test_filename, train_ratio)
    normalise_sentences(db, train_filename, normalised_sentences_filename)
    #collect_bayes_factors(db, normalised_sentences_filename)
    #collect_ngrams(db, normalised_sentences_filename)
    #calculate_ngram_factors(db)
    db.commit()
    db.close()

# NOTE: http://effbot.org/pyfaq/why-doesnt-python-release-the-memory-when-i-delete-a-large-object.htm
#
# "... "For speed", Python maintains an internal free list for integer objects. Unfortunately, that
#  free list is both immortal and unbounded in size. floats also use an immortal & unbounded free list..."
#
# So there is no way to get rid of integers and free up space for others, except
# for running in a subprocess. Well, at least we could make it parallel if it
# wasn't the RAM that is the bottleneck.


##############################################################################
# MAIN - Training the model
#

#train_input("test.db", "sample")
#train_input("dict.db", "final/en_US/all")
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="news"))
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="twitter"))


##############################################################################
# Using the model
#

num_words = None
words = None
sdrow = None
word_odds = None  # fallback guesses based on word frequencies only
words_sorted = None
qGetBayes = None
qGetPrefix = None
qGetNgram = None

def get_suggestions(db, prefix_words, partial=""):
    ''' Get the suggestion list for a normalised sentence '''
    global num_words, words, sdrow, word_odds, words_sorted
    global qGetBayes, qGetPrefix, qGetNgram

    log.info(u"Searching; prefix='{p}', hint='{h}'".format(p=" ".join(map(lambda x: words[x], prefix_words)), h=partial))

    # get the Bayesian relatives factors
    bayes_factors = {}
    # first chain the 'odds-factor of `i` occuring, provided that `w` occured'
    for w in prefix_words:
        #log.debug("Searching Bayesian relatives for; word='{w}'".format(w=words[w]))
        qGetBayes.execute("SELECT conditional, factor FROM bayes_t WHERE condition=?1", (w, ))
        for row in qGetBayes.fetchall():
            i = int(row[0])
            if (not i in words) or (i in prefix_words):
                continue
            # combine the factors by chaining (multiplying) them
            if row[1] is None:
                # NOTE: this means that word `i` occurs *only* in such sentences that also contain word `w`
                #bayes_factors[i] = float("infinity")
                pass
            elif not i in bayes_factors:
                bayes_factors[i] = word_odds[i] * float(row[1])  # fix it up with the base 'odds of `i` occuring'
            else:
                bayes_factors[i] *= float(row[1])

    # find the longest prefix that matches the end of prefix_words
    prefix_id = -1
    for j in xrange(len(prefix_words) - 1, -1, -1):
        w = prefix_words[j]
        qGetPrefix.execute("SELECT id FROM prefix_t WHERE parent=?1 AND word=?2", (prefix_id, w))
        result = qGetPrefix.fetchone()
        if not result:
            #log.debug("Prefix; id={i}, prefix='{p}'".format(i=prefix_id, p=" ".join(map(lambda x: words[x], prefix_words[j+1:]))))
            break
        prefix_id = int(result[0])
    #else:
        #log.debug("Prefix; id={i}, prefix='{p}'".format(i=prefix_id, p=" ".join(map(lambda x: words[x], prefix_words))))

    # get the n-grams that start with that prefix
    result = []
    qGetNgram.execute("SELECT follower, factor, occurences FROM ngram_t WHERE prefix=?1", (prefix_id, ))
    for row in qGetNgram.fetchall():
        i = int(row[0])
        if not i in words:
            log.error("Unknown ngram follower; i={i}".format(i=i))
            continue
        if not words[i].startswith(partial):
            continue
        occurences = int(row[2])
        f = float(row[1]) if row[1] is not None else 1e100

        # combine the Bayesian factors into the ngram-based ones
        if i in bayes_factors:
            f *= bayes_factors[i]
            del bayes_factors[i]

        #log.debug("N-gram candidate; id='{i}', word='{w}', ngram_factor='{f}', occurences='{n}'".format(i=i, w=words[i].encode("utf-8"), f=f, n=occurences))
        m = occurences * f / (1 + f)
        if not math.isnan(m):
            result.append((i, m))

    # we need them ordered by descending expected value
    result.sort(key=lambda x: -x[1])

    # if there are not enough suggestions, top up from the remaining Bayesian factors
    # NOTE: there is no 'occurences of condition' here, so sorting by odds factor instead
    if len(result) < N_sug:
        # listify the Bayesian results, because we want them sorted by odds
        if partial == "":
            bayes_factors = [ (i, f) for i, f in bayes_factors.iteritems() ]
        else:
            bayes_factors = [ (i, f) for i, f in bayes_factors.iteritems() if words[i].startswith(partial) ]
        bayes_factors.sort(key=lambda x: -x[1])
        result.extend(bayes_factors[: N_sug - len(result)])

    # if still too few suggestions, top up from the global words
    # NOTE: there is no 'occurences of condition' here, so sorting by raw occurence counter
    if len(result) < N_sug:
        for w in words_sorted:
            if len(result) >= N_sug:
                break;
            if not words[w[0]].startswith(partial):
                continue
            result.append(w)

    return [ (words[x[0]], x[1]) for x in result[:N_sug] ]


def test_input(db, f):
    global num_words, words, sdrow, word_odds, words_sorted
    global qGetBayes, qGetPrefix, qGetNgram

    qGetBayes = db.cursor()
    qGetPrefix = db.cursor()
    qGetNgram = db.cursor()

    q = db.cursor() 
    q.execute("SELECT id, word, occurences FROM word_t")
    word_odds = q.fetchall()  # [ (id1, word1, occurences1), (id2, word2, occurences2), ... ]
    q.close()
    word_odds = filter(lambda x: x[1] != "^" and x[1] != ",", word_odds) # FIXME: remove
    num_words = len(word_odds)
    words = dict((w[0], w[1]) for w in word_odds)  # { id1: word1, id2: word2, ... }
    sdrow = dict((w[1].lower(), w[0]) for w in word_odds)  # { word1: id1, word2: id2, ... }
    words_sorted = [ (w[0], w[2]) for w in word_odds ]
    words_sorted.sort(key=lambda x: -x[1])
    total_words = sum(w[2] for w in word_odds)
    word_odds = dict((w[0], float(w[2]) / (total_words - w[2])) for w in word_odds)

    total_found = 0
    total = 0
    keystrokes = 0
    while True:
        line = f.readline()
        if not line:
            break
        sentence = " ".join(split_to_sentences(line.lower().rstrip("\r\n"))).split(" ")
        if not sentence or not sentence[0]:
            continue
        log.debug(u"Sentence; normalised='{s}'".format(s=sentence))
        sentence_words = map(lambda w: sdrow[w] if w in sdrow else -1, sentence)

        for plen in xrange(0, len(sentence_words)):
            shouldbe = sentence[plen]
            log.debug(u"Next word; should_be='{s}'".format(s=shouldbe))
            hint = ""

            total += 1
            while hint != shouldbe:
                suggestions = get_suggestions(db, filter(lambda w: w >= 0, sentence_words[:plen]), partial=hint)
                log.debug(u"Result; suggestions='{s}'".format(s=list((x[0], round(x[1], 2)) for x in suggestions)))
                
                found = False
                i = 0
                for s in suggestions:
                    #log.debug(u"Suggestion; i='{i}', word='{w}', m='{m}'".format(i=i, w=words[s[0]], m=s[1]))
                    if s[0] == shouldbe:
                        log.debug("Found at {i} after a hint of {n}".format(i=i, n=len(hint)))
                        found = True
                        break
                    i += 1

                if found:
                    # got it, the user taps on the suggestion
                    keystrokes += 1
                    if hint == "":
                        total_found += 1
                    break

                if hint == shouldbe:
                    # completely missed it, the user typed in the word, and now taps to accept it
                    log.debug("Missed");
                    keystrokes += 1
                    break;

                # not found yet, the user taps on the next character
                keystrokes += 1
                hint += shouldbe[len(hint)]


            log.info("Keystrokes = {k}, success ratio = {r} = {f} / {t}".format(k=keystrokes, r=float(total_found)/total, f=total_found, t=total))

    qGetBayes.close()
    qGetPrefix.close()
    qGetNgram.close()

       

testfile = codecs.open("sample.test.txt", mode="r", encoding="utf-8")
tests = StringIO("""
The guy in front of me just bought a pound of bacon , a bouquet , and a case of
You're the reason why I smile everyday. Can you follow me please? It would mean the
Hey sunshine , can you follow me and make me the
Very early observations on the Bills game Offense still struggling but the
Go on a romantic date at the
Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my
Ohhhhh PointBreak is on tomorrow
Love that film and haven't seen it in quite some
After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little
Be grateful for the good times and keep the faith during the
If this isn't the cutest thing you've ever seen, then you must be""")

(db, already_exists) = open_dict_db("dict.db")
db.execute("PRAGMA query_only = ON")
test_input(db, tests)
#test_input(db, testfile)

testfile.close()

# vim: set et ts=4 sw=4:
