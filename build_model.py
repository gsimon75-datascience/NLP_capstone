#!/usr/bin/env python
import re
import os
import os.path
import logging
from time import time
import random
import codecs
import sqlite3

##############################################################################
# Parameters
#

# Number of suggestions
N_sug = 5

# Discard the least frequent percentage of words
p_discard = 0.1
p_keep = 1 - p_discard

# Maximal size of the suggestion dictionary
N_max = 200000

# Maximal prefix N-gram size
N_max_prefix_size = 4

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

if "PYTHON_LOGLEVEL" in os.environ:
    log.setLevel(int(os.environ["PYTHON_LOGLEVEL"]))
else:
    log.setLevel(logging.INFO)


##############################################################################
# Parts of the training process
#


# Normalisation regexes for whole lines - may produce line breaks
line_rules = [
    # pre-compiled regex                                                        replacement
    #
    # revert unicode punctuation to ascii
    [re.compile(u"[\u2018-\u201b\u2032\u2035`]"),                               "'"],
    [re.compile(u"[\u201c-\u201f\u2033\u2034\u2036\u2037\u2039\u203a\u2057\xab\xbb]"), "\""],
    [re.compile(u"[\u2010-\u2015\u2043]"),                                      "-"],
    [re.compile(u"[\u2024\u2027]"),                                             "."],
    [re.compile(u"\u2025"),                                                     ".."],
    [re.compile(u"\u2026"),                                                     "..."],
    [re.compile(u"[\u2000-\u200d\u2060\u202f\u205f]+"),                         " "],
    [re.compile(u"\u2063"),                                                     ","],
    [re.compile(u"\u2052"),                                                     "%"],
    [re.compile(u"[\u204e\u2055\u2062]"),                                       "*"],
    [re.compile(u"\u2052"),                                                     "%"],
    [re.compile(u"\u2064"),                                                     "+"],
    # sub-sentence separators to comma or newline with trailing-leading spaces
    [re.compile("\\r"),                                                         ""],
    [re.compile("^|$"),                                                         " "],
    [re.compile("(,\\s*)+"),                                                    " , "],
    # coalesce adjacent subsentence delimiters
    [re.compile("(,\\s*)+"),                                                    " , "],
    # strip leading and trailing subsentence delimiters
    [re.compile("(,\\s*)+$"),                                                   ""],
    [re.compile("^(\\s*,)+"),                                                   ""],
    # finally: split at sentence delimiters
    [re.compile("[.!?]+"),                                                      " \n "]
]

# Normalisation regexes for sub-sentences
sentence_rules = [
    # pre-compiled regex                                                        replacement
    # (NOTE: must be here at front, others rely on its result)
    # some common typos and abbreviations (NOTE: standalone only)
    [re.compile(" i "),                                                         " I "],
    [re.compile(" o' ", re.I),                                                  " of "],
    [re.compile(" ol' ", re.I),                                                 " old "],
    # not-in-word apostrophes, quotes, parentheses, underscores to space
    [re.compile(" '|' |[()\":/_]"),                                             " "],
    # zap all single characters except 'I' and 'a'
    [re.compile(" [^IAa,] "),                                                   " "],
    # remove only numbers
    [re.compile(" [-0-9, ]*[0-9][-0-9, ]* "),                                   " "],
    # zap all words without at least one lowercase letter
    # NOTE: no way to handle ALL CAPS WORDS correctly: proper nouns should be like 'John',
    # the rest should be lowercase, and we don't want to suggest all caps later, so we
    # can't keep it that way either
    [re.compile(" [A-Z0-9]+ "),                                                 " _ "],
    [re.compile(" [^a-z,]+ "),                                                  " "],
    # more than one dashes are delimiter
    [re.compile("-{2,}"),                                                       " "],
    # zap all words with invalid characters (valid: alnum, ', _, +, -)
    [re.compile(" [^ ]*[^ a-zA-Z0-9_'+-,][^ ]* "),                              " _ "],
    # zap non-alnums in front of words
    [re.compile(" [^a-zA-Z0-9,]+"),                                             " "],
    # zap non-alnums at the end of words
    [re.compile("[^a-zA-Z0-9,]+ "),                                             " "],
    # coalesce adjacent subsentence delimiters
    [re.compile("(,\\s*)+"),                                                    " , "],
    # coalesce whitespaces
    [re.compile("\\s+"),                                                        " "]
]

# after adding 'unknown word' marker, pull together such markers if adjacent
coalesce_unknowns = re.compile("_[ _]*_")

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
    if not already_exists:
        # freshly created, create tables and indices
        q = db.cursor()
        q.execute('''
        CREATE TABLE words_t (
         word       TEXT,
         termfreq   REAL,
         coverage   REAL,
         occurences INTEGER)''')
        q.execute("CREATE INDEX words_word_i ON words_t (word)")
        q.execute("CREATE INDEX words_occurences_i ON words_t (occurences)")

        q.execute('''
        CREATE TABLE dict_t (
         prefix     TEXT,
         suggestion TEXT,
         value      INTEGER)''')
        q.execute("CREATE INDEX dict_prefix_i ON dict_t (prefix)")
        q.execute("CREATE INDEX dict_value_i ON dict_t (value)")

        q.close()
        db.commit()

    return (db, already_exists)


def open_ngram_db(db_filename):
    ''' Open the SQLite database for the N-grams, creating and initialising it if it doesn't yet exist'''
    log.info("Using ngram db; db_filename='{f}'".format(f=db_filename))
    already_exists = os.path.isfile(db_filename)
    db = sqlite3.connect(db_filename)
    if not already_exists:
        # freshly created, create tables and indices
        q = db.cursor()
        q.execute("DROP INDEX IF EXISTS ngram_i")
        q.execute("DROP TABLE IF EXISTS ngram_t")
        q.execute('''
        CREATE TABLE ngram_t (
         prefix     TEXT,
         follower   TEXT,
         occurences INTEGER)''')
        q.execute("CREATE INDEX ngram_i ON ngram_t (prefix, follower)")
        q.close()
        db.commit()

    return (db, already_exists)


def get_wordset(db_dict):
    ''' Get the global word statistics '''
    qGlobalWordList = db_dict.cursor()
    qGlobalWordList.execute("SELECT word, termfreq, coverage, occurences FROM words_t ORDER BY occurences DESC LIMIT :nsug", {"nsug": N_sug})
    sorted_words = map(lambda rec: {"word": rec[0], "termfreq": rec[1], "coverage": rec[2], "occurences": rec[3]},
                       qGlobalWordList.fetchall())
    qGlobalWordList.close()
    return sorted_words


def collect_all_sentences(db_dict, input_filename, output_filename):
    '''Normalise a raw corpus and split into sub-sentences'''
    log.info("Collecting sentences; input='{i}', output='{o}'".format(i=input_filename, o=output_filename))
    if os.path.isfile(output_filename):
        return get_wordset(db_dict) if db_dict is not None else None

    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(output_filename, mode="w", encoding="utf-8")
    total_lines = 0
    all_words = {}
    for line in infile:
        for sentence in split_to_sentences(line):
            outfile.write(sentence + "\n")

            # count the words
            for word in sentence.split(" "):
                if word in all_words:
                    all_words[word] = 1 + all_words[word]
                else:
                    all_words[word] = 1

        total_lines += 1
        if need_progress_printout():
            log.debug("  Collected; lines='{l}'".format(l=total_lines))

    log.info("  Collected; lines='{l}'".format(l=total_lines))
    outfile.close()
    infile.close()

    # calculate the total number of words (needed for discarding the rarest ones)
    total = 0
    sorted_words = []
    for word, occurences in all_words.iteritems():
        total = total + occurences
        sorted_words.append({"occurences": occurences, "word": word})
    del all_words
    sorted_words.sort(key=lambda rec: rec["occurences"], reverse=True)

    cumulative = 0
    total = float(total)
    for idx in range(0, len(sorted_words)):
        cumulative = cumulative + sorted_words[idx]["occurences"]
        sorted_words[idx]["termfreq"] = sorted_words[idx]["occurences"] / total
        sorted_words[idx]["coverage"] = cumulative * 1.0 / total

    # store the global words in the dict database
    if db_dict is not None:
        q = db_dict.cursor()
        q.execute("DELETE FROM words_t")
        for word in sorted_words:
            q.execute("INSERT INTO words_t (word, termfreq, coverage, occurences) VALUES (:word, :tf, :cov, :occurences)",
                      {"word": word["word"], "tf": word["termfreq"], "cov": word["coverage"], "occurences": word["occurences"]})
        q.close()
        db_dict.commit()

    return sorted_words


def collect_common_sentences(input_filename, output_filename, words):
    '''Collect the sentences that contain only the p_keep most frequent words'''
    log.info("Filtering sentences; input='{i}', output='{o}'".format(i=input_filename, o=output_filename))
    # cut the word list to the top-P-percent
    words = filter(lambda rec: rec["coverage"] <= p_keep, words)

    if os.path.isfile(output_filename):
        return words

    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(output_filename, mode="w", encoding="utf-8")
    wordset = set(map(lambda rec: rec["word"], words))
    total_lines = 0
    for sentence in infile:
        # split and replace the rare words with '_'
        common_words = map(lambda word: word if word in wordset else "_", sentence.rstrip("\n").split(" "))
        if common_words:
            # pull together multiple adjacent '_'-s (if any)
            common_sentence = coalesce_unknowns.sub("_", " ".join(common_words))
            outfile.write("^ {s}\n".format(s=common_sentence))

        total_lines += 1
        if need_progress_printout():
            log.debug("  Filtered; lines='{l}'".format(l=total_lines))

    log.info("  Filtered; lines='{l}'".format(l=total_lines))
    outfile.close()
    infile.close()
    return words


def collect_ngrams(db_ngram, input_filename, n):
    ''' Collect the n-grams from the given corpus '''
    log.info("Collecting ngrams; infile='{i}', n='{n}'".format(i=input_filename, n=n))
    infile = open(input_filename, "r")
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
                        qNewNgram.execute('INSERT INTO ngram_t (prefix, follower, occurences) VALUES (:prefix, :follower, 1)', record)

        total_lines += 1
        if need_progress_printout():
            log.debug("  N-gram gen; n='{n}', lines='{l}'".format(n=n, l=total_lines))
    log.info("  N-gram gen; n='{n}', lines='{l}'".format(n=n, l=total_lines))
    db_ngram.commit()
    qIncNgramCounter.close()
    qNewNgram.close()
    infile.close()


def DL_cost(s, t, i, j, v0, v1, v2):
    ''' Damerau-Levenshtein cost function '''
    # D-L-dist cost function: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Definition
    # s, t are the strings; i, j are the current matrix position; v0, v1, v2 are the current, the previous,
    # and the one-before-previous matrix lines
    delCost = v0[j + 1] + 1
    insCost = v1[j] + 1
    sbsCost = v0[j] if s[i] == t[j] else v0[j] + 1
    swpCost = (v2[j - 1] + 1) if (i > 0) and (j > 0) and (s[i] == t[j - 1]) and (s[i - 1] == t[j]) else 999999
    return min(delCost, insCost, sbsCost, swpCost)


def DL_distance(s, t):
    ''' Damerau-Levenshtein distance of lists '''
    # This version uses 3 rows of storage, analoguous to the 2-row L-distance algorithm:
    # https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
    m = len(s)
    n = len(t)

    v0 = range(0, n + 1)
    v2 = []
    i = 0
    while True:
        v1 = [i + 1]
        for j in range(0, n):
            v1.append(DL_cost(s, t, i, j, v0, v1, v2))

        i += 1
        if i >= m:
            return v1[n]

        v2 = [i + 1]
        for j in range(0, n):
            v2.append(DL_cost(s, t, i, j, v1, v2, v0))

        i += 1
        if i >= m:
            return v2[n]

        v0 = [i + 1]
        for j in range(0, n):
            v0.append(DL_cost(s, t, i, j, v2, v0, v1))

        i += 1
        if i >= m:
            return v0[n]


def process_ngrams(db_ngram, db_dict, words):
    ''' Process the n-grams and update the dictionary with them '''
    log.info("Processing ngrams;")
    # get the top-N_sug words from the global distribution
    global_suggestion = map(lambda x: x["word"], words[0:N_sug])

    # get the current dictionary size, so we won't have to query it again and again
    q = db_dict.cursor()
    q.execute("SELECT COUNT(*), SUM(value) FROM dict_t")
    stats = q.fetchone()
    dict_stats = {"size": stats[0], "total_value": stats[1]}
    dict_stats["size"] = stats[0]
    dict_stats["total_value"] = 0 if stats[1] is None else stats[1]
    q.close()

    qCheapestDictItem = db_dict.cursor()
    qDiscardDictItem = db_dict.cursor()
    qNewDictItem = db_dict.cursor()
    qAddDictValue = db_dict.cursor()
    qParentPrefix = db_dict.cursor()

    def add_new_item(prefix, suggestion, value):
            qNewDictItem.execute("INSERT INTO dict_t (prefix, suggestion, value) VALUES (:prefix, :suggestion, :value)",
                                 {"prefix": prefix, "suggestion": " ".join(suggestion), "value": value})
            dict_stats["total_value"] = dict_stats["total_value"] + value
            dict_stats["size"] += 1

    def add_dict_value(prefix, value):
            qAddDictValue.execute("UPDATE dict_t SET value = value + :plusvalue WHERE prefix=:prefix",
                                  {"prefix": prefix, "plusvalue": value})
            dict_stats["total_value"] = dict_stats["total_value"] + value

    def find_parent_for(prefix):
        parent_suggestion = global_suggestion
        parent_prefix = None
        for start in range(1, len(prefix)):
            parent_prefix = " ".join(prefix_words[start:])
            qParentPrefix.execute("SELECT suggestion FROM dict_t WHERE prefix=:prefix", {"prefix": parent_prefix})
            if qParentPrefix.rowcount > 0:
                parent_suggestion = qParentPrefix.fetchone()[0].split(" ")
                break
        return (parent_prefix, parent_suggestion)

    # iterate through all prefixes
    qFollowers = db_ngram.cursor()
    qPrefixes = db_ngram.cursor()
    qPrefixes.execute("SELECT prefix, SUM(occurences) FROM ngram_t GROUP BY prefix")
    total_lines = 0
    while True:
        ngram = qPrefixes.fetchone()
        if ngram is None:
            break
        prefix = ngram[0]
        prefix_words = prefix.split(" ")
        occurences = ngram[1]

        # construct a suggestion list for this prefix
        qFollowers.execute("SELECT follower FROM ngram_t WHERE prefix=:prefix ORDER BY occurences DESC LIMIT :nsug",
                           {"prefix": prefix, "nsug": N_sug})
        suggestion = map(lambda x: x[0], qFollowers.fetchall())

        # find the suggestion list for the parent of this prefix, that is, for A-B-C it is
        # either of B-C, or of C, or if none exists in the dictionary, then the global one
        (parent_prefix, parent_suggestion) = find_parent_for(prefix)

        # calculate the value as occurences times distance from the parent suggestion
        value = occurences * DL_distance(suggestion, parent_suggestion)

        # update the dictionary
        if dict_stats["size"] < N_max:
            # if the dictionary is not full, then just insert this prefix and suggestion
            add_new_item(prefix, suggestion, value)
        else:
            # the dictionary is full, find the least valuable entry
            qCheapestDictItem.execute("SELECT prefix, value FROM dict_t ORDER BY value LIMIT 1")
            cheapest = qCheapestDictItem.fetchone()
            cheapest_prefix = cheapest[0]
            cheapest_value = cheapest[1]
            if value < cheapest_value:
                # this suggestion is not worth storing (its parents suggestion list will be offered instead)
                if parent_prefix is not None:
                    # however, this makes its parent more valuable
                    add_dict_value(parent_prefix, value)
            else:
                # this suggestion is worth storing, the cheapest will be discarded
                add_new_item(prefix, suggestion, value)
                # but it makes the parent of the cheapest one more valuable
                (parent_prefix, parent_suggestion) = find_parent_for(cheapest_prefix)
                if parent_prefix is not None:
                    add_dict_value(parent_prefix, cheapest_value)
                # discard the cheapest one
                qDiscardDictItem.execute("DELETE FROM dict_t WHERE prefix=:prefix", {"prefix": cheapest_prefix})
                dict_stats["total_value"] = dict_stats["total_value"] - cheapest_value
                dict_stats["size"] = dict_stats["size"] - 1

        total_lines += 1
        if need_progress_printout():
            log.debug("Dict update; lines='{l}', size='{s}', value='{v}'".format(l=total_lines, s=dict_stats["size"], v=dict_stats["total_value"]))
    log.info("Dict update; lines='{l}', size='{s}', value='{v}'".format(l=total_lines, s=dict_stats["size"], v=dict_stats["total_value"]))

    qPrefixes.close()
    qFollowers.close()
    qParentPrefix.close()
    qAddDictValue.close()
    qNewDictItem.close()
    qDiscardDictItem.close()
    qCheapestDictItem.close()
    db_dict.commit()


##############################################################################
# The training process
#

def train_input(dict_db_filename, basename):
    ''' The whole training process from original input to dictionary db '''
    orig_input_filename = "{b}.txt".format(b=basename)
    train_filename = "{b}.train.txt".format(b=basename)
    test_filename = "{b}.test.txt".format(b=basename)
    all_sentences_filename = "{b}.all.sentences".format(b=basename)
    common_sentences_filename = "{b}.common.sentences".format(b=basename)

    # open the dict database (global wordlist will be stored there as well)
    (db_dict, _) = open_dict_db(dict_db_filename)
    # split the input to 80% train + 20% test
    split_train_test(orig_input_filename, train_filename, test_filename, 0.8)
    # get the global words distribution
    words = collect_all_sentences(db_dict, train_filename, all_sentences_filename)
    # filter for the common words
    words = collect_common_sentences(all_sentences_filename, common_sentences_filename, words)

    for n in range(1, 1 + N_max_prefix_size):
        ngram_db_filename = "{b}.{n}.db".format(b=basename, n=n)
        (db_ngram, already_exists) = open_ngram_db(ngram_db_filename)
        if not already_exists:
            # build the n-grams
            collect_ngrams(db_ngram, common_sentences_filename, n)
        # update the dictionary with the n-grams
        process_ngrams(db_ngram, db_dict, words)
        db_ngram.close()

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
    common_sentences_filename = "test.common.sentences"

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
    collect_all_sentences(None, test_filename, all_sentences_filename)
    words = collect_common_sentences(all_sentences_filename, common_sentences_filename, words)

    # initialise the hit counters
    hit = [0 for i in range(N_sug)]
    miss = 0
    total_hit = 0

    infile = codecs.open(common_sentences_filename, mode="r", encoding="utf-8")
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

train_input("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="blogs"))
test_suggestions("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="blogs"), "test_results.csv")

# vim: set et ts=4 sw=4:
