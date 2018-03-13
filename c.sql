PRAGMA synchronous = OFF;
PRAGMA journal_mode = OFF;
PRAGMA secure_delete = OFF;
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA mmap_size = 4294967296;

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

-- INSERT OR REPLACE INTO ngram_t (prefix, follower, factor)
-- 	SELECT prefix, follower, ((total / prefix_occurences) - 1) / ((follower_occurences / occurences) - 1) AS factor
-- 	FROM ngram_t NATURAL JOIN follower_stat_t NATURAL JOIN prefix_stat_t;

INSERT OR REPLACE INTO ngram_t (prefix, follower, factor)
	SELECT prefix, follower, ((180228054 / prefix_occurences) - 1) / ((follower_occurences / occurences) - 1) AS factor
	FROM ngram_t NATURAL JOIN follower_stat_t NATURAL JOIN prefix_stat_t;
UPDATE ngram_t SET factor=1.79e308 WHERE factor IS NULL;
