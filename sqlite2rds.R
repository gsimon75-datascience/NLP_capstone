library(rlang)
library(tibble)
library(dplyr)
library(DBI)
#library(RSQLite)


db <- dbConnect(RSQLite::SQLite(), "dict.db")
dbExecute(db, "PRAGMA synchronous = OFF")
dbExecute(db, "PRAGMA secure_delete = OFF")
dbExecute(db, "PRAGMA locking_mode = EXCLUSIVE")
dbExecute(db, "PRAGMA mmap_size = 4294967296")
dbExecute(db, "PRAGMA cache_size = 20000")
dbExecute(db, "PRAGMA query_only = ON")

x <- dbReadTable(db, "word_t", row.names=FALSE)
saveRDS(x, "word_t.rds")

x <- dbReadTable(db, "bayes_t", row.names=FALSE)
saveRDS(x, "bayes_t.rds")

x <- dbReadTable(db, "prefix_t", row.names=FALSE)
saveRDS(x, "prefix_t.rds")

x <- dbReadTable(db, "ngram_t", row.names=FALSE)
saveRDS(x, "ngram_t.rds")


dbDisconnect(db)

#dbGetQuery(mydb, 'SELECT * FROM mtcars LIMIT 5')
#dbGetQuery(mydb, 'SELECT * FROM iris WHERE "Sepal.Length" < :x', params=list(x=4.6))
#
#rs <- dbSendQuery(mydb, 'SELECT * FROM mtcars')
#while (!dbHasCompleted(rs)) {
#	df <- dbFetch(rs, n = 10)
#	message(nrow(df))
#}
#dbClearResult(rs)
#
#rs <- dbSendQuery(mydb, 'SELECT * FROM iris WHERE "Sepal.Length" < :x')
#dbBind(rs, param = list(x = 4.5))
#nrow(dbFetch(rs))
#dbBind(rs, param = list(x = 4))
#nrow(dbFetch(rs))
#dbClearResult(rs)
#
#rs <- dbSendQuery(mydb, 'SELECT * FROM iris WHERE "Sepal.Length" = :x')
#dbBind(rs, param = list(x = seq(4, 4.4, by = 0.1)))
#nrow(dbFetch(rs))
#dbClearResult(rs)
#
#dbExecute(mydb, 'DELETE FROM iris WHERE "Sepal.Length" < 4')
#
#rs <- dbSendStatement(mydb, 'DELETE FROM iris WHERE "Sepal.Length" < :x')
#dbBind(rs, param = list(x = 4.5))
#dbGetRowsAffected(rs)
#dbClearResult(rs)
#
#dbDisconnect(mydb)
#

#valid_words <<- as.vector(rep(T, length(our_words$word)))
#names(valid_words) <<- our_words$word
# ... %>% Filter(f=function(x) !is.na(valid_words[x])) %>% ...

