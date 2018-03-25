library(rlang)
library(tibble)
#library(dplyr)
library(stringr)
library(tokenizers)
library(DBI)
#library(RSQLite)
library(plyr)

db <- dbConnect(RSQLite::SQLite(), "dict.db")
dbExecute(db, "PRAGMA synchronous = OFF")
dbExecute(db, "PRAGMA secure_delete = OFF")
dbExecute(db, "PRAGMA locking_mode = EXCLUSIVE")
dbExecute(db, "PRAGMA mmap_size = 4294967296")
dbExecute(db, "PRAGMA cache_size = 20000")
dbExecute(db, "PRAGMA query_only = ON")

words <- dbReadTable(db, "word_t", row.names=FALSE)
word_id_remap <- new.env(hash=T) # lookup from orig id to sequential
l_ply(1:nrow(words), function(x) { word_id_remap[[as.character(words$id[x])]] <- x })
words$id <- 1:nrow(words) # might as well zap it
saveRDS(words, "word_t.rds")
rm(words)

remap_word_ids <- function(l) { sapply(as.character(l), function(x) { word_id_remap[[x]] }) }

bayes <- dbReadTable(db, "bayes_t", row.names=FALSE)
bayes$condition <- remap_word_ids(bayes$condition)
bayes$conditional <- remap_word_ids(bayes$conditional)
saveRDS(bayes, "bayes_t.rds")
rm(bayes)

prefixes <- dbReadTable(db, "prefix_t", row.names=FALSE)
prefixes$word <- remap_word_ids(prefixes$word)
saveRDS(prefixes, "prefix_t.rds")
rm(prefixes)

ngrams <- dbReadTable(db, "ngram_t", row.names=FALSE)
ngrams$follower <- remap_word_ids(ngrams$follower)
saveRDS(ngrams, "ngram_t.rds")
rm(ngrams)

dbDisconnect(db)

