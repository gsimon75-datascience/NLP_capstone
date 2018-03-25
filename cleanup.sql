-- Run as `sqlite3 test.db '.read cleanup.sql'`

.output /dev/null
PRAGMA synchronous = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;
PRAGMA cache_size=20000;
.output
.echo on

-- remove Bayes pairs that refer to invalidated words
DELETE FROM bayes_t
WHERE (SELECT 42 FROM word_t WHERE id = bayes_t.condition LIMIT 1) IS NULL OR
      (SELECT 42 FROM word_t WHERE id = bayes_t.conditional LIMIT 1) IS NULL;

-- remove prefixes that refer to invalidated words or when no ngram uses them
DELETE FROM prefix_t WHERE (SELECT 42 FROM word_t WHERE id = prefix_t.word LIMIT 1) IS NULL;
DELETE FROM prefix_t WHERE (SELECT 42 FROM ngram_t WHERE id = ngram_t.prefix LIMIT 1) IS NULL;
DELETE FROM prefix_t WHERE parent != -1 AND (SELECT 42 FROM prefix_t parent WHERE parent.id = prefix_t.parent LIMIT 1) IS NULL;

-- remove ngram that refer to invalidated words or prefixes
DELETE FROM ngram_t
WHERE (SELECT 42 FROM prefix_t WHERE id = ngram_t.prefix LIMIT 1) IS NULL OR
      (SELECT 42 FROM word_t WHERE id = ngram_t.follower LIMIT 1) IS NULL;

-- remove words that noone refers to
DELETE FROM word_t
WHERE (SELECT 42 FROM bayes_t WHERE condition = word_t.id OR conditional = word_t.id LIMIT 1) IS NULL AND
      (SELECT 42 FROM prefix_t p WHERE p.word = word_t.id LIMIT 1) IS NULL AND
      (SELECT 42 FROM ngram_t n WHERE n.follower = word_t.id LIMIT 1) IS NULL;

SELECT 'words', COUNT(*) FROM word_t;
SELECT 'Bayesian pairs', COUNT(*) FROM bayes_t;
SELECT 'prefixes', COUNT(*) FROM prefix_t;
SELECT 'ngrams', COUNT(*) FROM ngram_t;

--VACUUM;
