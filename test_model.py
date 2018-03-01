#!/usr/bin/env python
import re
import os
import os.path
import logging
from time import time
import random
import codecs
import sqlite3
import sys

##############################################################################
# Parameters
#

# Number of suggestions
N_sug = 5

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
    log.setLevel(logging.DEBUG)


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



def get_wordset(db_dict):
    ''' Get the global word statistics '''
    qGlobalWordList = db_dict.cursor()
    #qGlobalWordList.execute("SELECT word, termfreq, coverage, occurences FROM words_t ORDER BY occurences DESC LIMIT :nsug", {"nsug": N_sug})
    qGlobalWordList.execute("SELECT word, termfreq, coverage, occurences FROM words_t ORDER BY occurences DESC")
    sorted_words = map(lambda rec: {"word": rec[0], "termfreq": rec[1], "coverage": rec[2], "occurences": rec[3]},
                       qGlobalWordList.fetchall())
    qGlobalWordList.close()
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



##############################################################################
# Using the model
#

def get_suggestions(db_dict, sentence, qLookupPrefix):
    ''' Get the suggestion list for a normalised sentence '''
    # NOTE: for large-scale testing performance a reuseable cursor can be provided
    use_local_cursor = qLookupPrefix is None
    if use_local_cursor:
        qLookupPrefix = db_dict.cursor()

    if type(sentence) == str:
        sentence = split_to_sentences(sentence)[0].split(" ")

    # try each line suffix as suggestion prefix

    for i in range(0, len(sentence)):
        prefix = " ".join(sentence[i:])
        qLookupPrefix.execute("SELECT suggestion FROM dict_t WHERE prefix=:prefix", {"prefix": prefix})
        result = qLookupPrefix.fetchone()
        if result is not None:
            log.debug("Found suggestions for prefix; p='{p}'".format(p=prefix))
            if use_local_cursor:
                qLookupPrefix.close()
            return result[0].split(" ")

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

#test_suggestions("dict.db", "final/{l}/{l}.{s}".format(l="en_US", s="blogs"), "test_results.csv")
#f#
# open the dict database
(db_dict, _) = open_dict_db("dict.db")
q=db_dict.cursor()
#q.execute("SELECT count(*) FROM dict_t")
#q.execute("SELECT count(*) from words_t")
#q.execute("SELECT word, termfreq, coverage, occurences FROM words_t ORDER BY occurences DESC")
#q.fetchall()
#log.debug("Looked up prefix; rowcount='{n}'".format(n=q.rowcount))


words = get_wordset(db_dict)
log.debug("Top words: {x}".format(x=" ".join(map(lambda rec: rec["word"], words[:20]))))
wordset = set(map(lambda rec: rec["word"], words))

default_suggestions = map(lambda x: x["word"], words)[:N_sug]

tests = [
          "The guy in front of me just bought a pound of bacon , a bouquet , and a case of",
          "You're the reason why I smile everyday",
          "Can you follow me please",
          "It would mean the",
          "Hey sunshine , can you follow me and make me the",
          "Very early observations on the Bills game Offense still struggling but the",
          "Go on a romantic date at the",
          "Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",
          "Ohhhhh PointBreak is on tomorrow",
          "Love that film and haven't seen it in quite some",
          "After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",
          "Be grateful for the good times and keep the faith during the",
          "If this isn't the cutest thing you've ever seen, then you must be"
]

for sentence in tests:
    log.info("Sentence: '{t}'".format(t=sentence))
    # split and replace the rare words with '_'
    common_words = map(lambda word: word if word in wordset else "_", split_to_sentences(sentence)[0].split(" "))
    # pull together multiple adjacent '_'-s (if any)
    common_sentence = coalesce_unknowns.sub("_", " ".join(common_words))

    q.execute("SELECT count(*) FROM dict_t")
    response = get_suggestions(db_dict, common_sentence, q)
    if response is None:
        response = default_suggestions
    log.info("Response: '{r}'".format(r=" ".join(response)))

db_dict.close()


# vim: set et ts=4 sw=4:
