PRAGMA synchronous = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;
PRAGMA cache_size=20000;
-- PRAGMA journal_mode = OFF;

--total = SELECT SUM(occurences) FROM word_t;
--total = 112955444

UPDATE bayes_t SET factor = (112955444.0 / (SELECT occurences FROM word_t WHERE id = bayes_t.conditional) - 1) / (1.0 * (SELECT occurences FROM word_t WHERE id=bayes_t.condition) / occurences - 1);

UPDATE ngram_t SET factor = (112955444.0 / (SELECT occurences FROM word_t WHERE id=ngram_t.follower) - 1) / (1.0 * (SELECT occurences FROM prefix_t WHERE id = ngram_t.prefix) / occurences - 1);

DELETE FROM bayes_t WHERE occurences < 2;


-- FIXUPS AFTER TRIMMING SOME WORDS - we have no 'ON DELETE CASCADE' here...

-- remove Bayes pairs that refer to invalidated words
DELETE FROM bayes_t
WHERE (SELECT 42 FROM word_t WHERE id = bayes_t.condition LIMIT 1) IS NULL OR
      (SELECT 42 FROM word_t WHERE id = bayes_t.conditional LIMIT 1) IS NULL;

-- remove prefixes that refer to invalidated words
DELETE FROM prefix_t WHERE (SELECT 42 FROM word_t WHERE id = prefix_t.word LIMIT 1) IS NULL;
DELETE FROM prefix_t WHERE parent != -1 AND (SELECT 42 FROM prefix_t parent WHERE parent.id = prefix_t.parent LIMIT 1) IS NULL;

DELETE FROM ngram_t
WHERE (SELECT 42 FROM prefix_t WHERE id = ngram_t.prefix LIMIT 1) IS NULL OR
      (SELECT 42 FROM word_t WHERE id = ngram_t.follower LIMIT 1) IS NULL;


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




