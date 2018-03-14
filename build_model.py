#!/usr/bin/env python
import re
import os
import os.path
import logging
from time import time
import random
import codecs
import sqlite3
import cPickle
import sys

##############################################################################
# Parameters
#

# Number of suggestions
N_sug = 4

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

        if level == self.num_keys:
            log.debug("  Committed; keys='{t}', cache_size='{n}'".format(t=self.progress, n=self.cache_size))
            self.db.commit()
            self.cache = {}
            self.cache_size = 0

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
            if self.cache_size >= self.cache_max_size:
                self.commit_cache()

    def close(self):
        self.commit_cache()
        self.qIncrEntry.close()
        self.qNewEntry.close()



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
    words = map(lambda rec: {"id": int(rec[0]), "word": rec[1]}, qGlobalWordList.fetchall())
    qGlobalWordList.close()
    return words


def normalise_sentences(db, input_filename, output_filename, collect_words=True):
    '''Normalise a raw corpus and split into sub-sentences'''

    if os.path.isfile(output_filename):
        log.info("Using normalised sentences; output='{o}'".format(o=output_filename))
        return get_wordset(db)

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

    words = [ {"id": 0, "word": "^"}, {"id":1, "word": ","} ] if collect_words else get_wordset(db)
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
                words.append({"id": i, "word": word})
                i += 1

        # store the global words in the dict database
        q = db.cursor()
        q.executemany("INSERT INTO word_t (id, word) VALUES (:id, :word)", words)
        q.close()
        db.commit()

    log.info("Normalising sentences; input='{i}', output='{o}'".format(i=temp_filename, o=output_filename))
    wordset = dict(map(lambda w: (w["word"].lower(), w["id"]), words))

    def normalise_caps(w):
        wl = w.lower()
        if wl in wordset:
            return wordset[wl]
        return -1

    infile = codecs.open(temp_filename, mode="r", encoding="utf-8")
    outfile = open(output_filename, mode="wb")
    total_lines = 0
    for sentence in infile:
        # split and replace the rare words with '_'
        sentence_words = filter(lambda i: i >= 0, map(lambda word: normalise_caps(word), sentence.rstrip("\n").split(" ")))
        if sentence_words:
            cPickle.dump(sentence_words, outfile, -1)

        total_lines += 1
        if need_progress_printout():
            log.debug("  Normalised; lines='{l}'".format(l=total_lines))

    log.info("  Normalised; lines='{l}'".format(l=total_lines))
    outfile.close()
    infile.close()
    os.unlink(temp_filename)
    return words


def collect_bayes_factors(db, input_filename, words):
    ''' Collect the Bayes-factor matrix '''

    N = len(words)
    log.info("Collecting Bayes factors; infile='{i}', num_words='{n}'".format(i=input_filename, n=N))
    if N <= 0:
        log.critical("No known words, you should rebuild the word db;")
        sys.exit(1)

    # iterate through the sentences and occurences the words that occur in the same sentence
    qBayesCounter = CounterUpdater(16777216, db, "bayes_t", "occurences", "condition", "conditional")
    infile = open(input_filename, "rb")
    total = 0
    total_lines = 0
    try:
        while True:
            w = cPickle.load(infile)
            nw = len(w)
            for i in range(nw - 1):
                iidx = w[i]
                for j in range(i + 1, nw):
                    jidx = w[j]
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
    qBayesFactors = CounterUpdater(16777216, db, "bayes_t", "factor", "condition", "conditional")
    q = db.cursor()
    q.execute("SELECT condition, SUM(occurences) FROM bayes_t GROUP BY condition")
    word_counts = dict((int(x[0]), float(x[1])) for x in q.fetchall())
    total = sum(word_counts.values())
    N = len(word_counts)
    for i, ci in word_counts.iteritems():
        if need_progress_printout():
            log.debug("  Bayes; row='{i}', of='{n}'".format(i=i, n=N))
        total_ci_m1 = (float(total) / ci) - 1
        q.execute("SELECT conditional, occurences FROM bayes_t WHERE condition=:i", {"i": i})
        for rec in q.fetchall():
            j = rec[0]
            occurences = rec[1]
            cj = word_counts[j]
            if cj != occurences:
                factor = total_ci_m1 / ((cj / occurences) - 1)
                qBayesFactors.increment(factor, i, j)
            # NOTE: NULL means Inf, will read as None

    qBayesFactors.close()
    q.close()
    db.commit()


def collect_ngrams(db, input_filename, n):
    ''' Collect the n-grams from the given corpus '''
    log.info("Collecting ngrams; infile='{i}'".format(i=input_filename))
    infile = open(input_filename, "rb")

    # NOTE: small caches, we'll need the ram for the prefixes
    qFollowerCounter = CounterUpdater(1048576, db, "word_t", "occurences", "id")
    qNgramCounter = CounterUpdater(1048576, db, "ngram_t", "occurences", "prefix", "follower")
    
    # NOTE: when recording a prefix, the following steps are needed:
    # 1. if it exists, its occurence count must be increased
    # 2. if not, then a new entry must be created
    # 1+2. in both cases we need its id, because it'll be used as parent for the extended prefixes
    # this latter requirement makes normal caching impossible - either each step must
    # be committed alone, or everything must be cached

    # `prefixes` is the cache of prefix_t in a tree representation:
    # { word: { "id": id, "word": word_id, "occurences": occurence_count, "children": { ...} } }
    prefixes = {}
    next_prefix_id = 1
    total_lines = 0
    try:
        while True:
            t = cPickle.load(infile)
            nt = len(t)
            pmax = min(n, nt - 1)
            for f in range(1, nt):
                follower = t[f]
                qFollowerCounter.increment(1, follower)
                p = prefixes
                for i in range(1, pmax + 1):
                    w = t[f - i]
                    if not w in p:
                        p[w] = { "id": next_prefix_id, "occurences": 1, "children": {} }
                        next_prefix_id += 1
                    qNgramCounter.increment(1, p[w]["id"], follower)
                    p = p[w]["children"]

            total_lines += 1
            if need_progress_printout():
                log.debug("  N-gram gen; lines='{l}', prefixes='{p}'".format(l=total_lines, p=next_prefix_id))
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
            qCommitPrefixes.execute("INSERT INTO prefix_t (id, parent, word, occurences) VALUES (:id, :parent, :word, :occurences)",
                                    {"id": rec["id"], "parent": parent_id, "word": word, "occurences": rec["occurences"]})
            total_lines[0] += 1
            if need_progress_printout():
                log.debug("  Prefix commit; done='{t}', total='{n}'".format(t=total_lines[0], n=next_prefix_id))
            commit_prefixes(rec["id"], rec["children"])
            rec["children"] = None

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
        SELECT n.prefix, n.follower, n.occurences, ((:total / pfx.occurences) - 1) / ((flw.occurences / n.occurences) - 1) AS factor
        FROM ngram_t n INNER JOIN word_t flw ON n.follower = flw.id INNER JOIN prefix_t pfx ON n.prefix = pfx.id''', {"total": total} )
    log.debug("Calculated N-gram Bayes factors; total='{t}', rows='{r}'".format(t=total, r=q.rowcount))
    db.commit()


#def DL_cost(s, t, i, j, v0, v1, v2):
#    ''' Damerau-Levenshtein cost function '''
#    # D-L-dist cost function: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Definition
#    # s, t are the strings; i, j are the current matrix position; v0, v1, v2 are the current, the previous,
#    # and the one-before-previous matrix lines
#    delCost = v0[j + 1] + 2
#    insCost = v1[j] + 2
#    sbsCost = v0[j] if s[i] == t[j] else v0[j] + 3
#    swpCost = (v2[j - 1] + 1) if (i > 0) and (j > 0) and (s[i] == t[j - 1]) and (s[i - 1] == t[j]) else 999999
#    return min(delCost, insCost, sbsCost, swpCost)
#
#
#def DL_distance(s, t):
#    ''' Damerau-Levenshtein distance of lists '''
#    # This version uses 3 rows of storage, analoguous to the 2-row L-distance algorithm:
#    # https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
#    m = len(s)
#    n = len(t)
#
#    v0 = range(0, n + 1)
#    v2 = []
#    i = 0
#    while True:
#        v1 = [i + 1]
#        for j in range(0, n):
#            v1.append(DL_cost(s, t, i, j, v0, v1, v2))
#
#        i += 1
#        if i >= m:
#            return v1[n]
#
#        v2 = [i + 1]
#        for j in range(0, n):
#            v2.append(DL_cost(s, t, i, j, v1, v2, v0))
#
#        i += 1
#        if i >= m:
#            return v2[n]
#
#        v0 = [i + 1]
#        for j in range(0, n):
#            v0.append(DL_cost(s, t, i, j, v2, v0, v1))
#
#        i += 1
#        if i >= m:
#            return v0[n]
#
#
#def process_ngrams(db, db, words):
#    ''' Process the n-grams and update the dictionary with them '''
#    log.info("Processing ngrams;")
#    # get the top-N_sug words from the global distribution
#    global_suggestion = map(lambda x: x["word"], words[0:N_sug])
#
#    # get the current dictionary size, so we won't have to query it again and again
#    q = db.cursor()
#    q.execute("SELECT COUNT(*), SUM(value) FROM dict_t")
#    stats = q.fetchone()
#    dict_stats = {"size": stats[0], "total_value": stats[1]}
#    dict_stats["size"] = stats[0]
#    dict_stats["total_value"] = 0 if stats[1] is None else stats[1]
#    q.close()
#
#    qCheapestDictItem = db.cursor()
#    qDiscardDictItem = db.cursor()
#    qNewDictItem = db.cursor()
#    qAddDictValue = db.cursor()
#    qParentPrefix = db.cursor()
#
#    def add_new_item(prefix, suggestion, value):
#            qNewDictItem.execute("INSERT INTO dict_t (prefix, suggestion, value) VALUES (:prefix, :suggestion, :value)",
#                                 {"prefix": prefix, "suggestion": " ".join(suggestion), "value": value})
#            dict_stats["total_value"] = dict_stats["total_value"] + value
#            dict_stats["size"] += 1
#
#    def add_dict_value(prefix, value):
#            qAddDictValue.execute("UPDATE dict_t SET value = value + :plusvalue WHERE prefix=:prefix",
#                                  {"prefix": prefix, "plusvalue": value})
#            dict_stats["total_value"] = dict_stats["total_value"] + value
#
#    def find_parent_for(prefix):
#        parent_suggestion = global_suggestion
#        parent_prefix = None
#        for start in range(1, len(prefix)):
#            parent_prefix = " ".join(prefix_words[start:])
#            qParentPrefix.execute("SELECT suggestion FROM dict_t WHERE prefix=:prefix", {"prefix": parent_prefix})
#            if qParentPrefix.rowcount > 0:
#                parent_suggestion = qParentPrefix.fetchone()[0].split(" ")
#                break
#        return (parent_prefix, parent_suggestion)
#
#    # iterate through all prefixes
#    qGetFollowers = db.cursor()
#    qPrefixes = db.cursor()
#    qPrefixes.execute("SELECT prefix, SUM(occurences) FROM ngram_t GROUP BY prefix")
#    total_lines = 0
#    while True:
#        ngram = qPrefixes.fetchone()
#        if ngram is None:
#            break
#        prefix = ngram[0]
#        prefix_words = prefix.split(" ")
#        occurences = ngram[1]
#
#        # construct a suggestion list for this prefix
#        qGetFollowers.execute("SELECT follower FROM ngram_t WHERE prefix=:prefix ORDER BY occurences DESC LIMIT :nsug",
#                           {"prefix": prefix, "nsug": N_sug})
#        suggestion = map(lambda x: x[0], qGetFollowers.fetchall())
#
#        # find the suggestion list for the parent of this prefix, that is, for A-B-C it is
#        # either of B-C, or of C, or if none exists in the dictionary, then the global one
#        (parent_prefix, parent_suggestion) = find_parent_for(prefix)
#
#        # calculate the value as occurences times distance from the parent suggestion
#        value = occurences * DL_distance(suggestion, parent_suggestion)
#
#        # update the dictionary
#        if dict_stats["size"] < N_max:
#            # if the dictionary is not full, then just insert this prefix and suggestion
#            add_new_item(prefix, suggestion, value)
#        else:
#            # the dictionary is full, find the least valuable entry
#            qCheapestDictItem.execute("SELECT prefix, value FROM dict_t ORDER BY value LIMIT 1")
#            cheapest = qCheapestDictItem.fetchone()
#            cheapest_prefix = cheapest[0]
#            cheapest_value = cheapest[1]
#            if value < cheapest_value:
#                # this suggestion is not worth storing (its parents suggestion list will be offered instead)
#                if parent_prefix is not None:
#                    # however, this makes its parent more valuable
#                    add_dict_value(parent_prefix, value)
#            else:
#                # this suggestion is worth storing, the cheapest will be discarded
#                add_new_item(prefix, suggestion, value)
#                # but it makes the parent of the cheapest one more valuable
#                (parent_prefix, parent_suggestion) = find_parent_for(cheapest_prefix)
#                if parent_prefix is not None:
#                    add_dict_value(parent_prefix, cheapest_value)
#                # discard the cheapest one
#                qDiscardDictItem.execute("DELETE FROM dict_t WHERE prefix=:prefix", {"prefix": cheapest_prefix})
#                dict_stats["total_value"] = dict_stats["total_value"] - cheapest_value
#                dict_stats["size"] = dict_stats["size"] - 1
#
#        total_lines += 1
#        if need_progress_printout():
#            log.debug("Dict update; lines='{l}', size='{s}', value='{v}'".format(l=total_lines, s=dict_stats["size"], v=dict_stats["total_value"]))
#    log.info("Dict update; lines='{l}', size='{s}', value='{v}'".format(l=total_lines, s=dict_stats["size"], v=dict_stats["total_value"]))
#
#    qPrefixes.close()
#    qGetFollowers.close()
#    qParentPrefix.close()
#    qAddDictValue.close()
#    qNewDictItem.close()
#    qDiscardDictItem.close()
#    qCheapestDictItem.close()
#    # db.commit()



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
    words = normalise_sentences(db, train_filename, normalised_sentences_filename)
    collect_bayes_factors(db, normalised_sentences_filename, words)
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
# The testing process
#

def test_suggestions(dict_db_filename, basename, result_csv_name):
    ''' The whole training process from original input to dictionary db '''
    test_filename = "{b}.test.txt".format(b=basename)
    normalised_sentences_filename = "test.normalised.sentences"

    log.info("Testing suggestions; basename='{b}', result='{r}'".format(b=basename, r=result_csv_name))
    # open the dict database
    (db, _) = open_dict_db(dict_db_filename)
    q = db.cursor()
    q.execute("PRAGMA query_only = ON")
    q.close()

    global qGetFollowers, qGetBayes, N, total_words, sdrow
    qGetFollowers = db.cursor()
    qGetBayes = db.cursor()
    # NOTE: In normal operation we would normalise the typed sentence
    # and get suggestions at its end
    # Now for testing we normalise the test sentenes, and get suggestions
    # for all their word boundaries.

    # normalise the test file
    words = normalise_sentences(db, test_filename, normalised_sentences_filename, False)
    N = len(words)
    total_words = sum(w["occurences"] for w in words)
    sdrow = dict((words[i]["word"], i + 1) for i in range(N))

    # initialise the hit counters
    hit = [0 for i in range(N_sug)]
    miss = 0
    total_hit = 0

    infile = codecs.open(normalised_sentences_filename, mode="r", encoding="utf-8")
    total_lines = 0
    log.info("Checking suggestions;")
    for sentence in infile:
        sentence_words = sentence.split(" ")

        for w in range(2, len(sentence_words) - 1):
            real_word = sentence_words[w]
            if real_word == "," or real_word == "_":
                continue

            log.info("Checking sentence: s='{s}', expected='{e}'".format(s=" ".join(sentence_words[:w]), e=real_word))
            suggestions = get_suggestions(db, words, sentence_words[:w])
            if suggestions is None:
                suggestions = []

            miss += 1  # book as missed by default
            for i in range(len(suggestions)):
                if real_word == suggestions[i]:
                    miss -= 1
                    hit[i] += 1
                    total_hit += 1

        total_lines += 1
        if need_progress_printout():
            log.debug("  Checked; lines='{l}', hit='{h}', miss='{m}'".format(l=total_lines, h=total_hit, m=miss))

    log.info("  Checked; lines='{l}', hit='{h}', miss='{m}'".format(l=total_lines, h=total_hit, m=miss))
    infile.close()
    db.close()

    # write the output file
    outfile = codecs.open(result_csv_name, mode="w", encoding="utf-8")
    outfile.write("\"tries\", \"occurences\"\n")
    outfile.write("\"0\", \"{c}\"\n".format(c=miss))
    for i in range(0, len(hit)):
        outfile.write("\"{idx}\", \"{c}\"\n".format(idx=1 + i, c=hit[i]))
    outfile.close()


##############################################################################
# MAIN
#

#train_input("test.db", "sample")
train_input("/mnt/dict.db", "final/en_US/all")
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="news"))
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="twitter"))

#test_suggestions("dict.db", "sample", "yadda.csv")

# vim: set et ts=4 sw=4:
