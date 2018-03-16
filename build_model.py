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
from psutil import _psutil_linux as pslin
import gc

##############################################################################
# Parameters
#

# Number of suggestions
N_sug = 4

# Maximal size of the suggestion dictionary
N_max = 2000000

# Maximal prefix N-gram size
N_max_prefix_size = 3

# 90% training, 10% testing
train_ratio = 1.0

##############################################################################
# Logging
#

log = logging.getLogger("Main")
formatter = logging.Formatter('%(asctime)s.%(msecs).03d - %(name)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')

file_handler = logging.FileHandler("build.log", mode="w", encoding="UTF8")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stderr_handler = logging.StreamHandler()
stderr_handler.setFormatter(formatter)
log.addHandler(stderr_handler)

if "NLP_DEBUG" in os.environ:
    log.setLevel(int(os.environ["NLP_DEBUG"]))
else:
    log.setLevel(logging.INFO)


##############################################################################
# Parts of the training process
#

# retain some memory. don't know how much, from how much total, and what for
# all mem-related stats have at least three different interpretations
min_free_mb = 1024

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
    [re.compile(" bday ", re.I),                            " birthday "],
    [re.compile(" dis ", re.I),                             " this "],
    [re.compile(" dat ", re.I),                             " that "],
    [re.compile(" i "),                                     " I "],
    [re.compile(" i'"),                                     " I'"],
    # some abbreviations (NOTE: standalone only)
    [re.compile(" c'mon ", re.I),                           " come on "],
    [re.compile(" c[ou]z ", re.I),                          " because "],
    [re.compile(" imma ", re.I),                            " I'm a "],
    [re.compile(" ofcoz ", re.I),                           " of course "],
    [re.compile(" pl[sz] ", re.I),                          " please "],
    [re.compile(" ppl ", re.I),                             " people "],
    [re.compile(" tho ", re.I),                             " though "],
    [re.compile(" u ", re.I),                               " you "],
    [re.compile(" ur ", re.I),                              " your "],
    [re.compile(" 2b ", re.I),                              " to be "],
    [re.compile(" o' ", re.I),                              " of "],
    [re.compile(" ol' ", re.I),                             " old "],
    # not-in-word apostrophes to space
    [re.compile(" '|' "),                                   " "],
    # zap all single characters except 'I' and 'a'
    [re.compile(" [^IAa,] "),                               " "],
    # zap all words with invalid characters (valid: alnum, ', _, +, -)
    #[re.compile(" [^ ]*[^, a-zA-Z0-9_'+-][^ ]* "),          " _ "],
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

progress_counter = 0
progress_delta = 5
def sql_progress():
    global progress_counter, progress_delta
    progress_counter += progress_delta
    log.debug("SQL progress; steps='{n}M'".format(n=progress_counter))

free_mb = 0
def near_oom():
    global free_mb
    free_mb = psutil.virtual_memory().available >> 20
    return free_mb <= min_free_mb


def dump_memstat():
    log.debug("psutil.virtual_memory() = {x}".format(x=psutil.virtual_memory()))
    log.debug("pslin.linux_sysinfo() = {x}".format(x=pslin.linux_sysinfo()))


def open_dict_db(db_filename):
    ''' Open the SQLite database for the dictionary, creating and initialising it if it doesn't yet exist'''
    log.info("Using dict db; db_filename='{f}'".format(f=db_filename))
    already_exists = os.path.isfile(db_filename)
    db = sqlite3.connect(db_filename)  # , isolation_level=None)
    global progress_delta
    db.set_progress_handler(sql_progress, progress_delta * 1000000)
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
            occurences      INTEGER NOT NULL DEFAULT 0)''')
        q.execute("CREATE INDEX word_word_i ON word_t (word)")

        q.execute('''
        CREATE TABLE bayes_t (
            condition   INTEGER NOT NULL,
            conditional INTEGER NOT NULL,
            occurences       INTEGER NOT NULL DEFAULT 0,
            factor      REAL,
            PRIMARY KEY (condition, conditional))''')
        # condition, conditional REFERENCES word_t

        q.execute('''
        CREATE TABLE prefix_t (
            id          INTEGER NOT NULL PRIMARY KEY,
            parent      INTEGER NOT NULL,
            word        INTEGER NOT NULL,
            occurences       INTEGER NOT NULL DEFAULT 0)''')
        q.execute("CREATE INDEX prefix_parent_i ON prefix_t (parent)")
        # parent REFERENCES prefix_t, word REFERENCES word_t

        q.execute('''
        CREATE TABLE ngram_t (
            prefix     INTEGER NOT NULL,
            follower   INTEGER NOT NULL,
            occurences      INTEGER NOT NULL DEFAULT 0,
            factor     REAL,
            PRIMARY KEY (prefix, follower))''')
        q.execute("CREATE INDEX ngram_follower_i ON ngram_t (follower)")
        # prefix REFERENCES prefix_t, follower REFERENCES word_t

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
                    condition=" AND ".join([ "{k}=?{n}".format(k=key_colnames[i], n=1+i) for i in range(self.num_keys) ]))
        self.sqlNewEntry = "INSERT INTO {table} ({keys}, {occurences}) VALUES ({placeholders}, ?{n})".format(
                    table=tablename,
                    keys=", ".join(key_colnames),
                    occurences=count_colname,
                    placeholders=", ".join([ "?{n}".format(n=1+i) for i in range(self.num_keys) ]),
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
                    log.error("Still near OOM; free_mb='{f}'".format(f=free_mb))
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
        line = rule[0].sub(rule[1], line.rstrip("\n"))

    result = []
    for sentence in line.split("\n"):
        for rule in sentence_rules:
            sentence = rule[0].sub(rule[1], sentence)
        sentence = sentence.strip()
        if sentence:
            result.append(u"^ {s}".format(s=sentence))

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
    wordset = set(["^", ","])
    i = 2
    for line in infile:
        for sentence in split_to_sentences(line):
            outfile.write(sentence + "\n")
            if not collect_words:
                continue
            for word in sentence.split(" "):
                wordset.add(word)
        total_lines += 1
        if need_progress_printout():
            log.debug("  Collected; lines='{l}'".format(l=total_lines))

    log.info("  Collected; lines='{l}'".format(l=total_lines))
    outfile.close()
    infile.close()

    words = [ (0, "^"), (1, ",") ] if collect_words else get_wordset(db)
    if collect_words:
        # coalesce the words that differ only in capitalisation: less caps letters wins
        wordset = map(lambda w: (w.lower(), w), wordset)
        wordset.sort()
        i = 0
        n = len(wordset) - 1
        while i < n:
            if wordset[i][0] == wordset[i + 1][0]:
                del wordset[i]
                n -= 1
            else:
                i += 1
        wordset = map(lambda w: w[1], wordset)

        indexed = set(["^", ","])
        i = 2
        for word in wordset:
            if not word in indexed:
                indexed.add(word)
                words.append((i, word))
                i += 1

        # store the global words in the dict database
        q = db.cursor()
        q.executemany("INSERT INTO word_t (id, word) VALUES (?1, ?2)", words)
        q.close()
        db.commit()

    log.info("Normalising sentences; input='{i}', output='{o}'".format(i=temp_filename, o=output_filename))
    words = dict(map(lambda w: (w[1].lower(), w[0]), words))

    infile = codecs.open(temp_filename, mode="r", encoding="utf-8")
    outfile = open(output_filename, mode="wb")
    total_lines = 0
    for sentence in infile:
        # split and replace the rare words with '_'
        sentence_words = map(lambda w: words[w], filter(lambda w: w in words, sentence.rstrip("\n").lower().split(" ")))
        if sentence_words:
            cPickle.dump(sentence_words, outfile, -1)

        total_lines += 1
        if need_progress_printout():
            log.debug("  Normalised; lines='{l}'".format(l=total_lines))

    log.info("  Normalised; lines='{l}'".format(l=total_lines))
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
            w = cPickle.load(infile)
            nw = len(w)
            for i in range(nw - 1):
                iidx = w[i]
                if iidx <= 1:
                    continue  # skip ^ and ,
                for j in range(i + 1, nw):
                    jidx = w[j]
                    if jidx <= 1:
                        continue  # skip ^ and ,
                    qBayesCounter.increment(1, iidx, jidx)
                    total += 1
                    if iidx != jidx:
                        qBayesCounter.increment(1, jidx, iidx)
                        total += 1

            total_lines += 1
            if need_progress_printout():
                log.debug("  Coincidence; total='{t}', lines='{l}'".format(t=total, l=total_lines))
    except EOFError:
        pass
    log.info("  Coincidence; total='{t}', lines='{l}'".format(t=total, l=total_lines))
    infile.close()
    qBayesCounter.close()

    # convert count(AB) terms to Bayes factor P(B|A) / P(B|#A)
    qBayesFactors = db.cursor();
    q = db.cursor()
    q.execute("SELECT condition, SUM(occurences) FROM bayes_t GROUP BY condition")
    word_counts = dict((int(x[0]), float(x[1])) for x in q.fetchall())
    total = float(sum(word_counts.values()))
    N = len(word_counts)
    for i, ci in word_counts.iteritems():
        if need_progress_printout():
            log.debug("  Bayes; row='{i}', of='{n}'".format(i=i, n=N))
        total_ci_m1 = (total / ci) - 1
        q.execute("SELECT conditional, occurences FROM bayes_t WHERE condition=?1", (i,))
        for rec in q.fetchall():
            j = rec[0]
            occurences = rec[1]
            cj = word_counts[j]
            if cj != occurences:
                qBayesFactors.execute("UPDATE bayes_t SET factor=?3 WHERE condition=?1 AND conditional=?2", (i, j, total_ci_m1 / ((cj / occurences) - 1)))
            # NOTE: NULL means Inf, will read as None

    qBayesFactors.close()
    q.close()
    db.commit()


def collect_ngrams(db, input_filename, n):
    ''' Collect the n-grams from the given corpus '''
    global free_mb
    log.info("Collecting ngrams; infile='{i}'".format(i=input_filename))
    infile = open(input_filename, "rb")

    # NOTE: small caches, we'll need the ram for the prefixes
    qFollowerCounter = CounterUpdater(0x1000000, db, "word_t", "occurences", "id")
    qNgramCounter = CounterUpdater(0x1000000, db, "ngram_t", "occurences", "prefix", "follower")
    
    # NOTE: when recording a prefix, the following steps are needed:
    # 1. if it exists, its occurence count must be increased
    # 2. if not, then a new entry must be created
    # 1+2. in both cases we need its id, because it'll be used as parent for the extended prefixes
    # this latter requirement makes normal caching impossible - either each step must
    # be committed alone, or everything must be cached

    # `prefixes` is the cache of prefix_t in a tree representation:
    # { word: [id, occurence_count, {children}], ... }
    prefixes = {}
    next_prefix_id = 1
    total_lines = 0
    try:
        while True:
            w = cPickle.load(infile)
            nw = len(w)
            for f in range(1, nw):
                follower = w[f]
                qFollowerCounter.increment(1, follower)
                pmax = min(n, f - 1)
                p = prefixes
                for i in range(1, pmax + 1):
                    word = w[f - i]
                    if not word in p:
                        p[word] = [ next_prefix_id, 1 ]
                        next_prefix_id += 1
                    else:
                        p[word][1] += 1
                    qNgramCounter.increment(1, p[word][0], follower)
                    if i < pmax:
                        if len(p[word]) < 3:
                            p[word].append({})
                        p = p[word][2]

            total_lines += 1
            if need_progress_printout():
                log.debug("  N-gram gen; lines='{l}', prefixes='{p}', mem='{m}'".format(l=total_lines, p=next_prefix_id, m=pslin.linux_sysinfo()))
    except EOFError:
        pass
    log.info("  N-gram gen; lines='{l}', prefixes='{p}'".format(l=total_lines, p=next_prefix_id))
    infile.close()
    qFollowerCounter.close()
    qNgramCounter.close()

    # commit `prefixes` to the db
    qCommitPrefixes = db.cursor()
    total_lines = [0]
    def commit_prefixes(parent_id, p):
        for word, rec in p.iteritems():
            qCommitPrefixes.execute("INSERT INTO prefix_t (id, parent, word, occurences) VALUES (?1, ?2, ?3, ?4)", (rec[0], parent_id, word, rec[1]))
            total_lines[0] += 1
            if need_progress_printout():
                log.debug("  Prefix commit; done='{t}', total='{n}'".format(t=total_lines[0], n=next_prefix_id))
            if len(rec) >= 3:
                commit_prefixes(rec[0], rec[2])
                rec[2] = None

    log.info("Committing prefixes; n='{n}'".format(n=next_prefix_id))
    commit_prefixes(0, prefixes)
    qCommitPrefixes.close()
    db.commit()

def calculate_ngram_factors(db):
    ''' Calculate Bayes factors for ngrams '''
    log.info("Calculating N-gram Bayes factors;")
    q = db.cursor()
    q.execute("SELECT SUM(occurences) FROM prefix_t")
    total = q.fetchone()[0]
    q.execute('''
    INSERT OR REPLACE INTO ngram_t (prefix, follower, occurences, factor)
        SELECT n.prefix, n.follower, n.occurences, ((?1 / pfx.occurences) - 1) / ((flw.occurences / n.occurences) - 1) AS factor
        FROM ngram_t n INNER JOIN word_t flw ON n.follower = flw.id INNER JOIN prefix_t pfx ON n.prefix = pfx.id''', (total,) )
    log.debug("Calculated N-gram Bayes factors; total='{t}', rows='{r}'".format(t=total, r=q.rowcount))
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
    #split_train_test(corpus_filename, train_filename, test_filename, train_ratio)
    #normalise_sentences(db, train_filename, normalised_sentences_filename)
    #collect_bayes_factors(db, normalised_sentences_filename)
    collect_ngrams(db, normalised_sentences_filename, N_max_prefix_size)
    calculate_ngram_factors(db)
    db.commit()
    db.close()


##############################################################################
# Using the model
#

qGetFollowers = None
qGetBayes = None
def get_followers_ML(u):
    ''' Maximum-Likelihood algorithm '''
    n = len(u)
    #log.debug("Check prefix; p='{p}'".format(p=" ".join(u)))
    qGetFollowers.execute("SELECT follower, occurences FROM ngram_t WHERE prefix=:prefix", {"prefix": " ".join(u)})
    wset = dict(map(lambda x: (x[0], int(x[1])), qGetFollowers.fetchall()))
    cu = 0
    t = 0
    for k, cuw in wset.iteritems():
        t += 1
        cu += cuw

    p = {}
    for w, cuw in wset.iteritems():
        p[w] = float(cuw) / cu
    return p


N = None
total_words = 0
sdrow = None
def get_suggestions(db, words, sentence):
    ''' Get the suggestion list for a normalised sentence '''
    if type(sentence) == str: sentence = sentence.split(" ")

    frow = [ 1 for i in range(N + 1) ]
    for word in sentence:
        if not word in sdrow:
            continue
        j = sdrow[word]
        qGetBayes.execute("SELECT conditional, factor FROM bayes_t WHERE condition=:j", {"j": j})
        for row in qGetBayes.fetchall():
            i = int(row[0])
            if row[1] is not None:
                frow[i] *= float(row[1])
            else:
                frow[i] = float("infinity")

    suggestions = []
    for i in range(0, len(sentence)):
        prefix = sentence[i:]

        ngram_results = get_followers_ML(prefix)
        if not ngram_results:
            continue

        log.info("  Prefix: {p}".format(p=" ".join(prefix)))
        suggestions = []
        for k, v in ngram_results.iteritems():
            p = v
            if k in sdrow:
                odds = frow[sdrow[k]]
                pbayes = odds / (1 + odds)
                #log.debug("Combining result; word='{w}', p_ngram='{pn}', p_bayes='{pb}'".format(w=k, pn=v, pb=pbayes))
                p *= pbayes
            suggestions.append((p, k))

        suggestions.sort(reverse=True)
        #return suggestions
        i = 0
        for x in suggestions:
            if x[0] == 0:
                break
            if i == N_sug:
                log.debug("  -----;")
            i += 1
            log.debug(u"  Guess; w='{w}', p='{p}'".format(w=x[1], p=x[0]))

        suggestions = map(lambda x: x[1], suggestions[:N_sug])
        break

    # if too few suggestions, top up from the Bayesian results
    #if len(suggestions) < N_sug:
    if False:
        bayes_guess = [ (frow[i] * words[i - 1]["occurences"] / total_words, i) for i in range(1, N + 1) ]
        bayes_guess.sort(reverse=True)
        i = 0
        while len(suggestions) < N_sug:
            if bayes_guess[i][0] == 0:
                break
            x = words[bayes_guess[i][1]]["word"]
            if not x in suggestions:
                log.debug(u"  Bayes guess; w='{w}'".format(w=x))
                suggestions.append(x)
            i += 1

    # if still too few suggestions, top up from the global words
    i = 0
    while len(suggestions) < N_sug:
        x = words[i]["word"]
        if not x in suggestions:
            log.debug(u"  Global guess; w='{w}'".format(w=x))
            suggestions.append(x)
        i += 1

    return suggestions


##############################################################################
# MAIN
#

#train_input("test.db", "sample")
train_input("/mnt/dict.db", "final/en_US/all")
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="news"))
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="twitter"))

# vim: set et ts=4 sw=4:
