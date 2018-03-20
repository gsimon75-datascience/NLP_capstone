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
from multiprocessing import Process
import gc
import bisect
import math

##############################################################################
log = logging.getLogger("Main")
#formatter = logging.Formatter('%(asctime)s.%(msecs).03d - %(name)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')
formatter = logging.Formatter('%(asctime)s - %(levelname)8s - %(message)s', datefmt='%H:%M:%S')
#formatter = logging.Formatter('%(levelname)8s - %(message)s', datefmt='%H:%M:%S')

file_handler = logging.FileHandler("split.log", mode="w", encoding="UTF8")
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

stderr_handler = logging.StreamHandler()
stderr_handler.setFormatter(formatter)
log.addHandler(stderr_handler)

if "NLP_DEBUG" in os.environ:
    log.setLevel(int(os.environ["NLP_DEBUG"]))
else:
    log.setLevel(logging.INFO)


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
        CREATE TABLE ngram_t (
            id          INTEGER NOT NULL,
            parent      INTEGER NOT NULL,
            word        INTEGER NOT NULL,
            occurences  INTEGER NOT NULL DEFAULT 0,
            factor      REAL,
            PRIMARY KEY (parent, word))''')
        q.execute("CREATE INDEX ngram_id_i ON ngram_t (id)")
        q.execute("CREATE INDEX ngram_word_i ON ngram_t (word)")

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



def collect_ngrams(db, input_filename):
    ''' Collect the n-grams from the given corpus '''
    log.info("Collecting ngrams; infile='{i}'".format(i=input_filename))
    infile = open(input_filename, "rb")

    # NOTE: In fact we are collecting the (n+1)-grams, which then will be
    # treated as a word following an n-gram, but that splitting will happen
    # only later, now they are just (n+1)-grams.

    # ngrams = [id, count, [child-words], [child-nodes]]
    ngrams = [-1, -1, [], []]
    last_ngram_id = 0
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
            for start in xrange(0, nw):
                nmax = min(N_max_prefix_size, nw - start)
                #log.debug("Loop of start; start={s}, nmax={n}".format(s=start, n=nmax))
                root = ngrams
                for word in w[start : start + nmax]:
                    idx = bisect.bisect_left(root[2], word)
                    #log.debug("Adding {x} to: (idx={i}, n={n})".format(x=word, i=idx, n=len(root[2])))
                    #dump(ngrams)
                    if idx != len(root[2]) and root[2][idx] == word:
                        root = root[3][idx]
                        root[1] += 1
                    else:
                        last_ngram_id += 1
                        newroot = [last_ngram_id, 1, [], []]
                        root[2].insert(idx, word)
                        root[3].insert(idx, newroot)
                        root = newroot

                    #log.debug("Result:")
                    #dump(ngrams)

                total_ngrams += nmax + 1

            total_lines += 1

            if need_progress_printout():
                log.debug("  N-gram gen; lines='{l}', ngrams='{n}', mem='{m}'".format(l=total_lines, n=total_ngrams, m=free_mb()))
    except EOFError:
        pass
    infile.close()

    log.warning("Committing N-grams to DB;")
    #dump(ngrams)
        
    qCommitNgrams = db.cursor()
    total_lines = [0]
    def commit_children_of(parent):
        for i in xrange(0, len(parent[2])):
            chword = parent[2][i]
            ch = parent[3][i]
            qCommitNgrams.execute("INSERT INTO ngram_t (id, parent, word, occurences) VALUES (?1, ?2, ?3, ?4)", (ch[0], parent[0], chword, ch[1]))
            total_lines[0] += 1
            if need_progress_printout():
                log.debug("  Prefix commit; done='{t}', total='{n}'".format(t=total_lines[0], n=last_ngram_id))
            commit_children_of(ch)
        del parent[:]

    log.info("Committing prefixes; n='{n}'".format(n=last_ngram_id))
    commit_children_of(ngrams)
    qCommitNgrams.close()
    db.commit()


##############################################################################

dict_db_filename = "dict.db"
#dict_db_filename = "test.db"
(db, already_exists) = open_dict_db(dict_db_filename)

q = db.cursor()
qGetChildren = db.cursor()
qCheckIfAlreadyThere = db.cursor()
qSimpleMove = db.cursor()
qMoveOccurences = db.cursor()
qNewNgram = db.cursor()

def get_children_of(parent_id):
    qGetChildren.execute("SELECT id, parent, word, occurences FROM prefix_t WHERE parent = ?1", (parent_id, ))
    return qGetChildren.fetchall()

total_lines = [0]
def move_under(pfx, new_parent, follower):
    new_prefix_id = pfx[0]
    if new_parent != pfx[1]:
        qCheckIfAlreadyThere.execute("SELECT id FROM prefix_t WHERE parent = ?1 AND word = ?2", (new_parent, pfx[2]))
        other_pfx = qCheckIfAlreadyThere.fetchone()
        if other_pfx is None:
            # no such prefix there yet, just move it
            qSimpleMove.execute("UPDATE prefix_t SET parent = ?1 WHERE id = ?2", (new_parent, pfx[0]))
        else:
            # melt into it
            qMoveOccurences.execute("UPDATE prefix_t SET occurences = occurences + ?1 WHERE id = ?2", (pfx[3], other_pfx[0]))
            new_prefix_id = other_pfx[0]

    total_lines[0] += 1
    if need_progress_printout():
        log.debug(" Processed; lines='{l}'".format(l=total_lines[0]))
    # move the children as well
    for child in get_children_of(pfx[0]):
        move_under(child, new_prefix_id, follower)

    qNewNgram.execute("INSERT INTO ngram_t (prefix, follower, occurences) VALUES (?1, ?2, ?3)", (new_prefix_id, follower, pfx[3]))


log.info("Listing followers;")
for flw in get_children_of(-1):
    log.info("Splitting follower; flw='{f}'".format(f=flw[0]))
    for pfx in get_children_of(flw[0]):
        #log.info("Splitting prefix; pfx='{p}'".format(p=pfx[0]))
        move_under(pfx, -2, flw[2])

q.execute("DELETE FROM prefix_t WHERE parent = -1")
q.execute("UPDATE prefix_t SET parent = -1 WHERE parent = -2")

qNewNgram.close()
qMoveOccurences.close()
qSimpleMove.close()
qCheckIfAlreadyThere.close()
qGetChildren.close()
q.close()

db.commit()
db.close()


# vim: set et ts=4 sw=4:
