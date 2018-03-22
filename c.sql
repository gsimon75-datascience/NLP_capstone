PRAGMA synchronous = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;
PRAGMA cache_size=20000;
-- PRAGMA journal_mode = OFF;

CREATE TABLE ngram_t (
	prefix     TEXT NOT NULL,
	follower   TEXT NOT NULL,
	occurences INTEGER NOT NULL DEFAULT 1,
	factor     REAL,
	PRIMARY KEY (prefix, follower));
CREATE INDEX ngram_follower_i ON ngram_t (follower);

CREATE TABLE bayes_t (
	condition   INTEGER NOT NULL,
	conditional INTEGER NOT NULL,
	occurences  INTEGER NOT NULL DEFAULT 1,
	factor      REAL,
	PRIMARY KEY (condition, conditional));

CREATE TABLE words_t (
	id         INTEGER NOT NULL PRIMARY KEY,
	word       TEXT NOT NULL,
	occurences INTEGER NOT NULL,
	termfreq   REAL,
	coverage   REAL);
CREATE INDEX words_word_i ON words_t (word);
CREATE INDEX words_occurences_i ON words_t (occurences);

CREATE TABLE follower_stat_t (
	follower   TEXT NOT NULL PRIMARY KEY,
	follower_occurences integer NOT NULL);
INSERT INTO follower_stat_t SELECT follower, SUM(occurences) AS follower_occurences FROM ngram_t GROUP BY follower;

CREATE TABLE prefix_stat_t (
	prefix   TEXT NOT NULL PRIMARY KEY,
	prefix_occurences integer NOT NULL);
INSERT INTO prefix_stat_t SELECT prefix, SUM(occurences) AS prefix_occurences FROM ngram_t GROUP BY prefix;

--------------------------------------------------------------------------------

INSERT OR REPLACE INTO ngram_t (prefix, follower, occurences, factor)
	SELECT prefix, follower, occurences, ((180228054 / prefix_occurences) - 1) / ((follower_occurences / occurences) - 1) AS factor
	FROM ngram_t NATURAL JOIN follower_stat_t NATURAL JOIN prefix_stat_t;
UPDATE ngram_t SET factor=1.79e308 WHERE factor IS NULL;


INSERT OR REPLACE INTO ngram_t (prefix, follower, occurences, factor)
	SELECT prefix, follower, occurences, ((180228054 / pfx.occurences) - 1) / ((flw.occurences / occurences) - 1) AS factor
	FROM ngram_t n INNER JOIN word_t flw ON n.follower = flw.id INNER JOIN prefix_t pfx ON n.prefix = pfx.id;
UPDATE ngram_t SET factor=1.79e308 WHERE factor IS NULL;

--------------------------------------------------------------------------------

ALTER TABLE ngram_t RENAME TO prefix_t;

UPDATE word_t SET occurences=(SELECT occurences FROM prefix_t WHERE word=word_t.id AND parent=-1);

--total = SELECT SUM(occurences) FROM ngram_t;
--total = 400143820
--UPDATE ngram_t SET factor = ((400143820.0 / (SELECT occurences FROM ngram_t parent WHERE parent.id = ngram_t.parent)) - 1) / ((1.0 * (SELECT occurences FROM word_t WHERE id = ngram_t.word) / occurences) - 1);


CREATE TABLE ngram_t (
	prefix		INTEGER NOT NULL,
	follower	INTEGER NOT NULL,
	occurences	INTEGER NOT NULL,
	factor		REAL,
	PRIMARY KEY (prefix, follower)
);

--total = SELECT SUM(occurences) FROM word_t;
--total = 112955444

UPDATE bayes_t SET factor = (112955444.0 / (SELECT occurences FROM word_t WHERE id = bayes_t.conditional) - 1) / (1.0 * (SELECT occurences FROM word_t WHERE id=bayes_t.condition) / occurences - 1);

UPDATE ngram_t SET factor = (112955444.0 / (SELECT occurences FROM word_t WHERE id=ngram_t.follower) - 1) / (1.0 * (SELECT occurences FROM prefix_t WHERE id = ngram_t.prefix) / occurences - 1);

DELETE FROM bayes_t WHERE occurences < 2;
