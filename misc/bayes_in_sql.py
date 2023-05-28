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

# Discard the least frequent percentage of words
p_discard = 0.1
p_keep = 1 - p_discard

# Maximal size of the suggestion dictionary
N_max = 2000000

# Maximal prefix N-gram size
N_max_prefix_size = 4

# 90% training, 10% testing
train_ratio = 0.9

##############################################################################
# Logging
#

log = logging.getLogger("Main")
formatter = logging.Formatter('%(asctime)s.%(msecs).03d - %(name)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')

file_handler = logging.FileHandler("build.log", mode="a", encoding="UTF8")
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
    [re.compile(" c'mon ", re.I),                           " come on "],
    [re.compile(" c[ou]z ", re.I),                          " because "],
    [re.compile(" dis ", re.I),                             " this "],
    [re.compile(" dat ", re.I),                             " that "],
    [re.compile(" i "),                                     " I "],
    [re.compile(" i'"),                                     " I'"],
    [re.compile(" imma ", re.I),                            " I'm a "],
    [re.compile(" ofcoz ", re.I),                           " of course "],
    [re.compile(" pl[sz] ", re.I),                          " please "],
    [re.compile(" ppl ", re.I),                             " people "],
    [re.compile(" tho ", re.I),                             " though "],
    [re.compile(" u ", re.I),                               " you "],
    [re.compile(" ur ", re.I),                              " your "],
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
            result.append(sentence)

    return result


def open_dict_db(db_filename):
    ''' Open the SQLite database for the dictionary, creating and initialising it if it doesn't yet exist'''
    log.info("Using dict db; db_filename='{f}'".format(f=db_filename))
    already_exists = os.path.isfile(db_filename)
    db = sqlite3.connect(db_filename)
    q = db.cursor()
    q.execute("PRAGMA synchronous = OFF")
    q.execute("PRAGMA journal_mode = OFF")
    q.execute("PRAGMA secure_delete = OFF")
    q.execute("PRAGMA locking_mode = EXCLUSIVE")
    q.execute("PRAGMA mmap_size = 4294967296")
    #q.execute("PRAGMA threads = 4")
    if not already_exists:
        # freshly created, create tables and indices
        q.execute('''
        CREATE TABLE words_t (
            id         INTEGER PRIMARY KEY,
            word       TEXT,
            occurences INTEGER,
            termfreq   REAL,
            coverage   REAL)''')
        q.execute("CREATE INDEX words_word_i ON words_t (word)")
        q.execute("CREATE INDEX words_occurences_i ON words_t (occurences)")

        #q.execute('''
        #CREATE TABLE stats_t (
        # name       TEXT,
        # value      INTEGER)''')

        q.execute('''
        CREATE TABLE ngram_t (
            prefix     TEXT,
            follower   TEXT,
            occurences INTEGER DEFAULT 1,
            factor     REAL)''')
        q.execute("CREATE INDEX ngram_prefix_follower_i ON ngram_t (prefix, follower)")
        q.execute("CREATE INDEX ngram_follower_i ON ngram_t (follower)")

        q.execute('''
        CREATE TABLE bayes_t (
            condition   INTEGER,
            conditional INTEGER,
            occurences  INTEGER DEFAULT 1,
            factor      REAL)''')
        q.execute("CREATE INDEX bayes_condition_conditional_i ON bayes_t (condition, conditional)")

        #q.execute('''
        #CREATE TABLE dict_t (
        # prefix     TEXT,
        # suggestion TEXT,
        # value      INTEGER)''')
        #q.execute("CREATE INDEX dict_prefix_i ON dict_t (prefix)")
        #q.execute("CREATE INDEX dict_value_i ON dict_t (value)")

    q.close()
    db.commit()
    return (db, already_exists)


def get_wordset(db_dict):
    ''' Get the global word statistics '''
    qGlobalWordList = db_dict.cursor()
    qGlobalWordList.execute("SELECT word, occurences, termfreq, coverage FROM words_t ORDER BY occurences DESC")
    sorted_words = map(lambda rec: {"word": rec[0], "occurences": rec[1], "termfreq": rec[2], "coverage": rec[3]},
                       qGlobalWordList.fetchall())
    qGlobalWordList.close()
    return sorted_words


def normalise_sentences(db_dict, input_filename, output_filename):
    '''Normalise a raw corpus and split into sub-sentences'''
    if os.path.isfile(output_filename):
        log.info("Using collected sentences; output='{o}'".format(o=output_filename))
        return get_wordset(db_dict) if db_dict is not None else None

    temp_filename = "{of}.temp".format(of=output_filename)
    log.info("Collecting sentences; input='{i}', output='{o}'".format(i=input_filename, o=temp_filename))
    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(temp_filename, mode="w", encoding="utf-8")
    total_lines = 0
    words = {}
    for line in infile:
        for sentence in split_to_sentences(line):
            outfile.write(sentence + "\n")

            # count the words
            for word in sentence.split(" "):
                if word == "," or word == "_":
                    pass
                elif word in words:
                    words[word] = 1 + words[word]
                else:
                    words[word] = 1

        total_lines += 1
        if need_progress_printout():
            log.debug("  Collected; lines='{l}'".format(l=total_lines))

    log.info("  Collected; lines='{l}'".format(l=total_lines))
    outfile.close()
    infile.close()

    # coalesce the words that differ only in capitalisation: less caps letters wins
    words = map(lambda w: { "word": w[0], "lower": w[0].lower(), "occurences": w[1]}, words.iteritems())
    words.sort(key=lambda w: (w["lower"], w["word"]))
    i = 0
    n = len(words) - 1
    while i < n:
        if words[i]["lower"] == words[i + 1]["lower"]:
            words[i + 1]["occurences"] += words[i]["occurences"]
            del words[i]
            n -= 1
        else:
            i += 1

    # calculate the total number of words (needed for discarding the rarest ones)
    total = sum(map(lambda w: w["occurences"], words))
    words.sort(key=lambda w: w["occurences"], reverse=True)
    cumulative = 0
    total = float(total)
    for i in range(0, len(words)):
        cumulative = cumulative + words[i]["occurences"]
        words[i]["termfreq"] = words[i]["occurences"] / total
        words[i]["coverage"] = cumulative * 1.0 / total

    # store the global words in the dict database
    if db_dict is not None:
        q = db_dict.cursor()
        q.execute("DELETE FROM words_t")
        q.executemany("INSERT INTO words_t (word, occurences, termfreq, coverage) VALUES (:word, :occurences, :termfreq, :coverage)", words)
        q.close()
        db_dict.commit()

    log.info("Normalising sentences; input='{i}', output='{o}'".format(i=temp_filename, o=output_filename))
    # cut the word list to the top-P-percent
    wordset = dict(map(lambda w: (w["word"].lower(), w["word"]), words))

    def normalise_caps(w):
        wl = w.lower()
        return wordset[wl] if wl in wordset else "_"

    infile = codecs.open(temp_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(output_filename, mode="w", encoding="utf-8")
    total_lines = 0
    for sentence in infile:
        # split and replace the rare words with '_'
        sentence_words = map(lambda word: normalise_caps(word), sentence.rstrip("\n").split(" "))
        if sentence_words:
            # pull together multiple adjacent '_'-s (if any)
            n = len(sentence_words) - 1
            i = 0
            while i < n:
                if sentence_words[i] == "_":
                    while i < n and sentence_words[i + 1] == "_":
                        del sentence_words[i + 1]
                        n -= 1
                i += 1

            outfile.write("^ {s}\n".format(s=" ".join(sentence_words)))

        total_lines += 1
        if need_progress_printout():
            log.debug("  Normalised; lines='{l}'".format(l=total_lines))

    log.info("  Normalised; lines='{l}'".format(l=total_lines))
    outfile.close()
    infile.close()
    os.unlink(temp_filename)
    return words


def collect_ngrams(db_ngram, input_filename, n):
    ''' Collect the n-grams from the given corpus '''
    log.info("Collecting ngrams; infile='{i}', n='{n}'".format(i=input_filename, n=n))
    infile = codecs.open(input_filename, "r", encoding="utf-8")
    qIncNgramCounter = db_ngram.cursor()
    qNewNgram = db_ngram.cursor()
    total_lines = 0
    for line in infile:
        t = line.rstrip("\n").split(" ")
        nt = len(t)
        if nt > n:
            for start in range(0, nt - n):
                if t[start + n] != "_" and t[start + n] != ",":
                    record = {"prefix": " ".join(t[start:(start + n)]), "follower": t[start + n]}
                    qIncNgramCounter.execute('UPDATE ngram_t SET occurences = occurences + 1 WHERE prefix=:prefix AND follower=:follower', record)
                    if qIncNgramCounter.rowcount <= 0:
                        qNewNgram.execute('INSERT INTO ngram_t (prefix, follower) VALUES (:prefix, :follower)', record)

        total_lines += 1
        if need_progress_printout():
            log.debug("  N-gram gen; n='{n}', lines='{l}'".format(n=n, l=total_lines))
    log.info("  N-gram gen; n='{n}', lines='{l}'".format(n=n, l=total_lines))
    db_ngram.commit()
    qIncNgramCounter.close()
    qNewNgram.close()
    infile.close()


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
#def process_ngrams(db_ngram, db_dict, words):
#    ''' Process the n-grams and update the dictionary with them '''
#    log.info("Processing ngrams;")
#    # get the top-N_sug words from the global distribution
#    global_suggestion = map(lambda x: x["word"], words[0:N_sug])
#
#    # get the current dictionary size, so we won't have to query it again and again
#    q = db_dict.cursor()
#    q.execute("SELECT COUNT(*), SUM(value) FROM dict_t")
#    stats = q.fetchone()
#    dict_stats = {"size": stats[0], "total_value": stats[1]}
#    dict_stats["size"] = stats[0]
#    dict_stats["total_value"] = 0 if stats[1] is None else stats[1]
#    q.close()
#
#    qCheapestDictItem = db_dict.cursor()
#    qDiscardDictItem = db_dict.cursor()
#    qNewDictItem = db_dict.cursor()
#    qAddDictValue = db_dict.cursor()
#    qParentPrefix = db_dict.cursor()
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
#    qFollowers = db_ngram.cursor()
#    qPrefixes = db_ngram.cursor()
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
#        qFollowers.execute("SELECT follower FROM ngram_t WHERE prefix=:prefix ORDER BY occurences DESC LIMIT :nsug",
#                           {"prefix": prefix, "nsug": N_sug})
#        suggestion = map(lambda x: x[0], qFollowers.fetchall())
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
#    qFollowers.close()
#    qParentPrefix.close()
#    qAddDictValue.close()
#    qNewDictItem.close()
#    qDiscardDictItem.close()
#    qCheapestDictItem.close()
#    db_dict.commit()


def collect_bayes_factors(input_filename, words):
    ''' Collect the Bayes-factor matrix '''

    if os.path.isfile("words.pck") and os.path.isfile("bayes_factors.pck"):
        return
    log.info("Collecting Bayes factors; infile='{i}'".format(i=input_filename))
    outfile = open("words.pck", "w")
    cPickle.dump(words, outfile)
    outfile.close()

    N = len(words)
    log.debug("Num of words; n='{n}'".format(n=N))
    if N <= 0:
        log.critical("No known words, you should rebuild the word db;")
        sys.exit(1)
    # create a word dict for converting word id
    sdrow = dict((words[i]["word"], i + 1) for i in range(N))

    qNewBayes = db_dict.cursor()
    qIncrBayes = db_dict.cursor()

    def increment_bayes(when, then):
        record = {"when": when, "then": then}
        qIncrBayes.execute("UPDATE bayes_t SET occurences = occurences + 1 WHERE condition=:when AND conditional=:then", record)
        if qIncrBayes.rowcount <= 0:
            qNewBayes.execute('INSERT INTO bayes_t (condition, conditional) VALUES (:when, :then)', record)

    # iterate through the sentences and count the words that occur in the same sentence
    infile = codecs.open(input_filename, "r", encoding="utf-8")
    total_lines = 0
    total = 0
    for line in infile:
        w = line.rstrip("\n").split(" ")
        nw = len(w)
        for i in range(nw - 1):
            if not w[i] in sdrow:
                continue
            iidx = sdrow[w[i]]
            for j in range(i + 1, nw):
                if not w[j] in sdrow:
                    continue
                jidx = sdrow[w[j]]

                increment_bayes(iidx, jidx)
                total += 1
                if iidx != jidx
                    increment_bayes(jidx, iidx)
                    total += 1

        total_lines += 1
        if need_progress_printout():
            log.debug("  Coincidence; total='{t}', lines='{l}'".format(t=total, l=total_lines))
    log.info("  Coincidence; total='{t}', lines='{l}'".format(t=total, l=total_lines))
    infile.close()

    qIncrBayes.close()
    qNewBayes.close()

    # convert count(AB) terms to Bayes factor P(B|A) / P(B|#A)
    q = db_dict.cursor()
    q.execute('''
        CREATE TABLE bayes_sums_t (
            id                  INTEGER PRIMARY KEY,
            count               INTEGER,
            complement_count    INTEGER)''')
    q.execute("INSERT INTO bayes_sums_t (id, count) SELECT condition, SUM(occurences) FROM bayes_t GROUP BY condition")
    q.execute("SELECT SUM(count) from bayes_sums_t")
    total_counts = int(q.fetchone[0])
    q.execute("UPDATE bayes_sums_t SET complement_count = :total - count", {"total": total_counts})

    q.execute("UPDATE bayes_t SET factor = (occurences / :ci) / ((:cj - occurences) / :c_not_i) WHERE condition=:i AND conditional=:j",

    word_counts = dict((int(x[0]), float(x[1])) for x in q.fetchall())
    total_count = sum(word_counts.values())
    N = len(word_counts)
    for i, ci in word_counts.iteritems():
        if need_progress_printout():
            log.debug("  Bayes; row='{i}', of='{n}'".format(i=i, n=N))
        c_not_i = total_count - ci
        for j, cj in word_counts.iteritems():
            q.execute("UPDATE bayes_t SET factor = (occurences / :ci) / ((:cj - occurences) / :c_not_i) WHERE condition=:i AND conditional=:j",
                      {"i": i, "j": j, "ci": ci, "cj": cj, "c_not_i": c_not_i})
            q.fetchone()
            # NOTE: div by 0 means Inf, stored as NULL, will read as None






##############################################################################
# The training process
#

def train_input(dict_db_filename, basename):
    ''' The whole training process from original input to dictionary db '''
    orig_input_filename = "{b}.txt".format(b=basename)
    train_filename = "{b}.train.txt".format(b=basename)
    test_filename = "{b}.test.txt".format(b=basename)
    normalised_sentences_filename = "{b}.normalised.sentences".format(b=basename)

    # open the dict database (global wordlist will be stored there as well)
    (db_dict, _) = open_dict_db(dict_db_filename)
    # split the input to 80% train + 20% test
    split_train_test(orig_input_filename, train_filename, test_filename, train_ratio)
    # get the global words distribution
    words = normalise_sentences(db_dict, train_filename, normalised_sentences_filename)

    collect_bayes_factors(normalised_sentences_filename, words)

    for n in range(2, 1 + N_max_prefix_size):
        ngram_db_filename = "{b}.{n}.db".format(b=basename, n=n)
        (db_ngram, already_exists) = open_ngram_db(ngram_db_filename)
        if not already_exists:
            # build the n-grams
            collect_ngrams(db_ngram, normalised_sentences_filename, n)
        # update the dictionary with the n-grams
        #process_ngrams(db_ngram, db_dict, words)
        db_ngram.close()
        #os.unlink(ngram_db_filename)

    # close the dict database
    db_dict.close()


##############################################################################
# Using the model
#

def get_suggestions(db_dict, sentence, qLookupPrefix=None):
    ''' Get the suggestion list for a normalised sentence '''
    # NOTE: for large-scale testing performance a reuseable cursor can be provided
    use_local_cursor = qLookupPrefix is None
    if use_local_cursor:
        qLookupPrefix = db_dict.cursor()

    if type(sentence) == str:
        sentence = sentence.split(" ")

    # try each line suffix as suggestion prefix
    for i in range(0, len(sentence)):
        qLookupPrefix.execute("SELECT suggestion FROM dict_t WHERE prefix=:prefix",
                              {"prefix": " ".join(sentence[i:])})
        log.debug("Looked up prefix; rowcount='{n}', p='{p}'".format(n=qLookupPrefix.rowcount, p=" ".join(sentence[i:])))
        if qLookupPrefix.rowcount > 0:
            suggestions = qLookupPrefix.fetchone()[0]
            if use_local_cursor:
                qLookupPrefix.close()
            return suggestions.split(" ")

    # tough luck, no prefix found, let the caller dig up the defaults
    return None


##############################################################################
# The testing process
#

def test_suggestions(dict_db_filename, basename, result_csv_name):
    ''' The whole training process from original input to dictionary db '''
    test_filename = "{b}.test.txt".format(b=basename)
    all_sentences_filename = "test.all.sentences"
    normalised_sentences_filename = "test.common.sentences"

    log.info("Testing suggestions; basename='{b}', result='{r}'".format(b=basename, r=result_csv_name))
    # open the dict database
    (db_dict, _) = open_dict_db(dict_db_filename)
    # get the global suggestions
    words = get_wordset(db_dict)
    default_suggestions = map(lambda x: x["word"], words)[:N_sug]

    # NOTE: In normal operation we would normalise the typed sentence
    # and get suggestions at its end
    #
    # Now for testing we normalise the test sentenes, and get suggestions
    # for all their word boundaries.

    # normalise the test file
    normalise_sentences(None, test_filename, all_sentences_filename)
    words = collect_common_sentences(all_sentences_filename, normalised_sentences_filename, words)

    # initialise the hit counters
    hit = [0 for i in range(N_sug)]
    miss = 0
    total_hit = 0

    infile = codecs.open(normalised_sentences_filename, mode="r", encoding="utf-8")
    total_lines = 0
    q = db_dict.cursor()  # reuseable cursor for performance
    log.info("Checking suggestions;")
    for sentence in infile:
        sentence_words = sentence.split(" ")

        for w in range(0, len(sentence_words) - 1):
            real_word = sentence_words[w + 1]

            suggestions = get_suggestions(db_dict, sentence[:w], q)
            if suggestions is None:
                suggestions = default_suggestions

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
    db_dict.close()

    # write the output file
    outfile = codecs.open(result_csv_name, mode="w", encoding="utf-8")
    outfile.write("\"tries\", \"count\"\n")
    outfile.write("\"0\", \"{c}\"\n".format(c=miss))
    for i in range(0, len(hit)):
    	outfile.write("\"{idx}\", \"{c}\"\n".format(idx=1 + i, c=hit[i]))
    outfile.close()


##############################################################################
# MAIN
#

train_input("dict.db", "final/en_US/all")
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="news"))
#train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="twitter"))
#test_suggestions("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="blogs"), "test_results.csv")

# vim: set et ts=4 sw=4:
