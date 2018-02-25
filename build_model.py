#!/usr/bin/env python
import sys
reload(sys)
# NOTE: utf8 is backwards compatible with ascii, so all code that hasn't produced
# "can't encode" exceptions with ascii, will work with utf8
sys.setdefaultencoding('UTF8')
import re
import os.path
from time import time
import codecs
import sqlite3
import cPickle

## Parameters

# Number of suggestions
N_sug = 5

# Discard the least frequent percentage of words
p_discard = 0.1
p_keep = 1 - p_discard

# Maximal size of the suggestion dictionary
N_max = 100000

locale = "en_US"
sources = ["blogs", "news", "twitter"]

# Print progress messages only once in a second
last_progress_printout = time()
def need_progress_printout():
    global last_progress_printout
    now = time()
    if (now - last_progress_printout) > 1:
        last_progress_printout = now
        return True
    return False

# Formulate a corpus base path from locale and source type
def basename(locale, src):
    return "final/{l}/{l}.{s}".format(l=locale, s=src)


# Normalisation regexes for whole sentences - may produce line breaks
sentence_rules = [
    # pre-compiled regex                                                        replacement
    #
    # revert unicode punctuation to ascii
    [re.compile(u"[\u2018-\u201b\u2032\u2035`]"),                               "'"],
    [re.compile(u"[\u201c-\u201f\u2033\u2034\u2036\u2037\u2039\u203a\u2057\xab\xbb]"),  "\""],
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
    # sub-sentence separators to newline with trailing-leading spaces
    [re.compile("[.,!?]"),                                                      " \n "]
]

# Normalisation regexes for sub-sentences
subsentence_rules = [
    # pre-compiled regex                                                        replacement
    # handle ^ and $ by whitespace rules, and word separator is also space
    # (NOTE: must be here at front, others rely on its result)
    [re.compile("^|$"),                                                         " "],
    # some common typos and abbreviations (NOTE: standalone only)
    [re.compile(" i "),                                                         " I "],
    [re.compile(" o' ", re.I),                                                  " of "],
    [re.compile(" ol' ", re.I),                                                 " old "],
    # not-in-word apostrophes, quotes, parentheses to space
    [re.compile(" '|' |[()\":/]"),                                              " "],
    # zap all words without at least one lowercase letter
    # NOTE: no way to handle ALL CAPS WORDS correctly: proper nouns should be like 'John',
    # the rest should be lowercase, and we don't want to suggest all caps later, so we
    # can't keep it that way either
    [re.compile(" [^a-z]+ "),                                                   " "],
    # more than one dashes are delimiter
    [re.compile("-{2,}"),                                                       " "],
    # zap all words with invalid characters (valid: alnum, ', _, +, -)
    [re.compile(" [^ ]*[^ a-zA-Z0-9_'+-][^ ]* "),                               " "],
    # zap all single characters except 'I' and 'a'
    [re.compile(" [^IAa] "),                                                    " "],
    # zap non-alnums in front of words
    [re.compile(" \\W+"),                                              " "],
    # zap non-alnums at the end of words
    [re.compile("\\W+ "),                                              " "],
    # coalesce whitespaces
    [re.compile("\\s+"),                                                        " "]
]

def collect_all_subsentences(basename):
    '''Normalise a raw corpus and split into sub-sentences'''
    input_filename = "{b}.txt".format(b=basename)
    output_filename = "{b}.all.subs".format(b=basename)
    words_filename = "{b}.words.pck".format(b=basename)

    if os.path.isfile(words_filename) and os.path.isfile(output_filename): 
        wordsfile = open(words_filename, "r")
        sorted_words = cPickle.load(wordsfile)
        wordsfile.close()
        return sorted_words

    print "Collecting subsentences; input='{i}', output='{o}'".format(i=input_filename, o=output_filename)
    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(output_filename, mode="w", encoding="utf-8")

    total_lines = 0
    all_words = {}
    for sentence in infile:
        for rule in sentence_rules:
            sentence = rule[0].sub(rule[1], sentence)

        for subsentence in sentence.split("\n"):
            for rule in subsentence_rules:
                subsentence = rule[0].sub(rule[1], subsentence)

            subsentence = subsentence.strip()
            if subsentence:
                # write the output
                outfile.write(subsentence + "\n")

                # count the words
                for word in subsentence.split(" "):
                    if word in all_words:
                        all_words[word] = 1 + all_words[word]
                    else:
                        all_words[word] = 1


        total_lines = total_lines + 1
        if need_progress_printout():
            print "  Processed block; lines='{l}'".format(l=total_lines)

    print "  Processed block; lines='{l}'".format(l=total_lines)
    outfile.close()
    infile.close()

    # calculate the total number of words (needed for discarding the rarest ones)
    total = 0
    sorted_words = []
    for word, count in all_words.iteritems():
        total = total + count
        sorted_words.append({"count": count, "word": word})
    del all_words
    sorted_words.sort(key=lambda rec: rec["count"], reverse=True)

    cumulative = 0
    total = float(total)
    for idx in range(0, len(sorted_words)):
        cumulative = cumulative + sorted_words[idx]["count"]
        sorted_words[idx]["termfreq"] = sorted_words[idx]["count"] / total
        sorted_words[idx]["coverage"] = cumulative * 1.0 / total

    wordsfile = open(words_filename, "w")
    cPickle.dump(sorted_words, wordsfile, -1)
    wordsfile.close()

    return sorted_words


def collect_common_subsentences(basename):
    '''Collect the subsentences that contain only the p_keep most frequent words'''
    input_filename = "{b}.all.subs".format(b=basename)
    output_filename = "{b}.common.subs".format(b=basename)

    # build and cut the word list to the top-P-percent
    words = filter(lambda rec: rec["coverage"] <= p_keep, collect_all_subsentences(basename))

    if os.path.isfile(output_filename): 
        return words

    q = db_dict.cursor()
    q.execute("DELETE FROM words_t")
    for word in words:
        q.execute("INSERT INTO words_t (word, occurences) VALUES (:word, :count)",
                  { "word": word["word"], "count": word["count"] })
    q.close()
    db_dict.commit()

    print "Filtering subsentences; input='{i}', output='{o}'".format(i=input_filename, o=output_filename)
    infile = codecs.open(input_filename, mode="r", encoding="utf-8")
    outfile = codecs.open(output_filename, mode="w", encoding="utf-8")

    wordset = set(map(lambda rec: rec["word"], words))
    total_lines = 0
    for sentence in infile:
        common_words = filter(lambda word: word in wordset, sentence.rstrip("\n").split(" "))
        if common_words:
            outfile.write("^ {s}\n".format(s=" ".join(common_words)))

        total_lines = total_lines + 1
        if need_progress_printout():
            print "  Processed block; lines='{l}'".format(l=total_lines)

    print "  Processed block; lines='{l}'".format(l=total_lines)
    outfile.close()
    infile.close()
    return words


def open_dict_db(dbfile):
    ''' Open the SQLite database for the dictionary, creating and initialising it if it doesn't yet exist'''
    need_create_tables = not os.path.isfile(dbfile) 
    db = sqlite3.connect(dbfile)
    if need_create_tables:
        q = db.cursor()
        q.execute('''
        CREATE TABLE words_t (
         word       TEXT,
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

    return db

def open_ngram_db(dbfile):
    ''' Open the SQLite database for the ngrams, creating and initialising it if it doesn't yet exist'''
    db = sqlite3.connect(dbfile)
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

    return db

db_dict = open_dict_db("dict.db")
db_ngram = None

def collect_ngrams(basename, n):
    ''' Collect the n-grams from the given corpus '''
    global db_ngram

    db_filename = "{b}.{n}.db".format(b=basename, n=n)
    input_filename = "{b}.common.subs".format(b=basename)
    
    # get the global words distribution
    words = collect_common_subsentences(basename)
    
    print "Collecting ngrams; infile='{i}', n='{n}'".format(i=input_filename, n=n) 

    # (re-)create the ngram database
    db_ngram = open_ngram_db(db_filename)

    infile = open(input_filename, "r")
    qIncNgramCounter = db_ngram.cursor()
    qNewNgram = db_ngram.cursor()
    total_lines = 0
    for line in infile:
        t = line.rstrip("\n").split(" ")
        nt = len(t)
        #print "Input; nt='{nt}', tokens='{t}'".format(nt=nt, t=" ".join(t)) 
        if nt > n:
            for start in range(0, nt - n):
                record = { "prefix": " ".join(t[start : (start + n)]), "follower": t[start + n] }
                #print "prefix='{p}', follower='{f}'".format(p=record["prefix"], f=record["follower"])
                qIncNgramCounter.execute('UPDATE ngram_t SET occurences = occurences + 1 WHERE prefix=:prefix AND follower=:follower', record)
                if qIncNgramCounter.rowcount <= 0:
                    qNewNgram.execute('INSERT INTO ngram_t (prefix, follower, occurences) VALUES (:prefix, :follower, 1)', record)

        total_lines = total_lines + 1
        if need_progress_printout():
            print "  Processed block; lines='{l}'".format(l=total_lines)
    print "  Processed block; lines='{l}'".format(l=total_lines)
    qIncNgramCounter.close()
    qNewNgram.close()
    db_ngram.commit()
    infile.close()
    return words


def DL_distance(s, t):
    ''' Damerau-Levenshtein distance of lists '''
    # This version uses 3 rows of storage, analoguous to the 2-row L-distance algorithm
    #
    # Iterative L-dist with 2 rows of storage: https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows
    # D-L-dist cost function: https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Definition
    m = len(s)
    n = len(t)

    v0 = range(0, n + 1)
    i = 0
    while True:
        v1 = [i + 1]
        for j in range(0, n):
            delCost = v0[j + 1] + 1
            insCost = v1[j] + 1
            sbsCost = v0[j] if s[i] == t[j] else v0[j] + 1
            swpCost = (v2[j - 1] + 1) if (i > 0) and (j > 0) and (s[i] == t[j - 1]) and (s[i - 1] == t[j]) else 999999
            v1.append(min(delCost, insCost, sbsCost, swpCost))

        i = i + 1
        if i >= m:
            return v1[n]

        v2 = [i + 1]
        for j in range(0, n):
            delCost = v1[j + 1] + 1
            insCost = v2[j] + 1
            sbsCost = v1[j] if s[i] == t[j] else v1[j] + 1
            swpCost = (v0[j - 1] + 1) if (i > 0) and (j > 0) and (s[i] == t[j - 1]) and (s[i - 1] == t[j]) else 999999
            v2.append(min(delCost, insCost, sbsCost, swpCost))

        i = i + 1
        if i >= m:
            return v2[n]

        v0 = [i + 1]
        for j in range(0, n):
            delCost = v2[j + 1] + 1
            insCost = v0[j] + 1
            sbsCost = v2[j] if s[i] == t[j] else v2[j] + 1
            swpCost = (v1[j - 1] + 1) if (i > 0) and (j > 0) and (s[i] == t[j - 1]) and (s[i - 1] == t[j]) else 999999
            v0.append(min(delCost, insCost, sbsCost, swpCost))

        i = i + 1
        if i >= m:
            return v0[n]


def process_ngrams(basename, n):
    ''' Process the n-grams and update the dictionary with them '''
    global db_ngram
    global db_dict

    # get the top-N_sug words from the global distribution
    global_suggestion = map(lambda x: x["word"], collect_ngrams(basename, n)[0:N_sug])
    print "global suggestion: {x}".format(x=global_suggestion)

    # get the current dictionary size, so we won't have to query it again and again
    q = db_dict.cursor()
    q.execute("SELECT COUNT(*), SUM(value) FROM dict_t")
    stats = q.fetchone()
    dict_stats = { "size": stats[0], "total_value": stats[1] }
    dict_stats["size"] = stats[0]
    dict_stats["total_value"] = 0 if stats[1] is None else stats[1]
    q.close()

    qCheapestDictItem = db_dict.cursor()
    qDiscardDictItem = db_dict.cursor()

    qNewDictItem = db_dict.cursor()
    def add_new_item(prefix, suggestion, value):
            qNewDictItem.execute("INSERT INTO dict_t (prefix, suggestion, value) VALUES (:prefix, :suggestion, :value)",
                                 { "prefix": prefix, "suggestion": " ".join(suggestion), "value": value })
            dict_stats["total_value"] = dict_stats["total_value"] + value
            dict_stats["size"] = dict_stats["size"] + 1

    qAddDictValue = db_dict.cursor()
    def add_dict_value(prefix, value):
            qAddDictValue.execute("UPDATE dict_t SET value = value + :plusvalue WHERE prefix=:prefix",
                                  { "prefix": prefix, "plusvalue": value })
            dict_stats["total_value"] = dict_stats["total_value"] + value

    qParentPrefix = db_dict.cursor()
    def find_parent_for(prefix):
        parent_suggestion = global_suggestion
        parent_prefix = None
        for start in range(1, len(prefix)):
            parent_prefix = " ".join(prefix_words[start:])
            qParentPrefix.execute("SELECT suggestion FROM dict_t WHERE prefix=:prefix", { "prefix": parent_prefix })
            if qParentPrefix.rowcount > 0:
                parent_suggestion = qParentPrefix.fetchone()[0].split(" ")
                break
        return (parent_prefix, parent_suggestion)

    # iterate through all prefixes
    qFollowers = db_ngram.cursor()
    qPrefixes = db_ngram.cursor()
    qPrefixes.execute("SELECT prefix, SUM(occurences) FROM ngram_t GROUP BY prefix");
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
                           { "prefix": prefix, "nsug": N_sug });
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
                # this suggestion is worth storing, the cheapest must be discarded
                add_new_item(prefix, suggestion, value)
                # but it makes the parent of the cheapest one more valuable
                (parent_prefix, parent_suggestion) = find_parent_for(cheapest_prefix)
                if parent_prefix is not None:
                    add_dict_value(parent_prefix, cheapest_value)
                # discard the cheapest one
                qDiscardDictItem.execute("DELETE FROM dict_t WHERE prefix=:prefix", { "prefix": cheapest_prefix })
                dict_stats["total_value"] = dict_stats["total_value"] - cheapest_value
                dict_stats["size"] = dict_stats["size"] - 1

        total_lines = total_lines + 1
        if need_progress_printout():
            print "Dict; lines='{l}', size='{s}', value='{v}'".format(l=total_lines, s=dict_stats["size"], v=dict_stats["total_value"])

    qPrefixes.close()
    qFollowers.close()
    qParentPrefix.close()
    qAddDictValue.close()
    qNewDictItem.close()
    qDiscardDictItem.close()
    qCheapestDictItem.close()
    db_dict.commit()


#collect_common_subsentences(basename(locale, "blogs"))
for n in range(1, 6):
    process_ngrams(basename(locale, "blogs"), n)

db_dict.close()
if db_ngram is not None:
    db_ngram.close()


#collect_subsentences("sample")
#words_blogs = collect_subsentences(basename(locale, "blogs")) %>% mutate(src="blogs")
#words_news = collect_subsentences(basename(locale, "news")) %>% mutate(src="news")
#words_twitter = collect_subsentences(basename(locale, "twitter")) %>% mutate(src="twitter")

# vim: set et ts=4 sw=4:
