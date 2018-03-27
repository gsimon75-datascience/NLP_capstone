-- Run as `sqlite3 test.db '.read cleanup.sql'`

.output /dev/null
PRAGMA synchronous = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;
PRAGMA cache_size=20000;
.output
.echo on

-- deleting prefixes by level
create table prefixlevel_t (id integer not null primary key, level integer not null);
create index prefixlevel_level_i on prefixlevel_t(level);
insert into prefixlevel_t select id, 1 from prefix_t where parent = -1;
insert into prefixlevel_t select id, 2 from prefix_t where (select level from prefixlevel_t where id=prefix_t.parent) == 1;
insert into prefixlevel_t select id, 3 from prefix_t where (select level from prefixlevel_t where id=prefix_t.parent) == 2;
insert into prefixlevel_t select id, 4 from prefix_t where (select level from prefixlevel_t where id=prefix_t.parent) == 3;

select level, count(id) from prefixlevel_t group by level order by level;
delete from prefix_t where (select level from prefixlevel_t where id=prefix_t.id) == 4;
drop table prefixlevel_t;

-- cutting away factors around 1
delete from ngram_t where factor is null or factor between 0.5 and 2; -- approx 14.7%, seems ok
delete from bayes_t where factor is null or factor between 0.04 and 25;  -- approx 14.7%, seems ok

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

-- final cleanup: remove occurences from bayes_t and prefix-by-id index
CREATE TABLE tmp_bayes_t (
	condition   INTEGER NOT NULL,
	conditional INTEGER NOT NULL,
	factor      REAL NOT NULL,
	PRIMARY KEY (condition, conditional)
);
INSERT INTO tmp_bayes_t (condition, conditional, factor) SELECT condition, conditional, factor FROM bayes_t;
DROP TABLE bayes_t;
ALTER TABLE tmp_bayes_t RENAME TO bayes_t;

DROP INDEX prefix_id_i;

VACUUM;
