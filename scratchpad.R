library(tibble)
library(dplyr)
library(tokenizers)


collect_subsentences <- function(input_basename) {
	input_filename <- paste0(input_basename, ".txt")
	output_filename <- paste0(input_basename, ".subs")
	words_filename <- paste0(input_basename, ".words.rds")

	all_words <- list()
	if (!file.exists(output_filename)) {

		input <- file(input_filename, open="r")
		output <- file(output_filename, open="w")
		lines_per_block <- 10000

		message("Collecting subsentences; input='", input_filename, "', output='", output_filename, "'")
		total_lines <- 0
		while (T) {
			lines <- readLines(con=input, n=lines_per_block, encoding="UTF-8", skipNul=T)
			num_lines = length(lines)
			if (num_lines <= 0) break

			lines <- lines %>%
			gsub(pattern="[ \t\u2000-\u200d\u2060\u202f\u205f]+", replacement=" ") %>%
			# here: whitespaces are coalesced into a single space
			gsub(pattern="[.,!?()\u2010-\u205e]", replacement="\n") %>%
			# here: sub-sentence delimiters are replaced with line break marks
			gsub(pattern="^\n+", replacement="") %>% gsub(pattern="\n+$", replacement="") %>% strsplit("\n+") %>% unlist(recursive=F) %>%
			# here: sub-sentences are split apart
			tokenize_word_stems(language="english")
			# here: words are replaced by list of their stems

			# count the words
			for (stemlist in lines) {
				for (word in stemlist) {
					if (is.null(all_words[[word]])) {
						all_words[[word]] <- 1
					} else {
						all_words[[word]] <- 1 + all_words[[word]]
					}
				}
			}

			# write the output
			lines %>% sapply(function(x) {paste0(x,collapse=" ")}) %>% writeLines(con=output)

			total_lines <- total_lines + num_lines
			message("  Processed block; lines='", total_lines, "'")
		}
		close(output)
		close(input)

		all_words <- rownames_to_column(data.frame(n=unlist(all_words)), var="word") %>% tibble::as_tibble()
		saveRDS(all_words, words_filename)
	} else {
		all_words <- readRDS(words_filename)
	}

	all_words
}

keep_most_frequent <- function(table, pct) {
	threshold <- pct * sum(table$n)
	table %>% arrange(desc(n)) %>% mutate(total=cumsum(n)) %>% filter(total<threshold) %>% mutate(total=NULL)
}

generate_ngrams_counted <- function(input_basename, n) { # NOTE: ram exhausted, can't do all in one step
	input_filename = paste0(input_basename, ".subs")
	output_filename = paste0(input_basename, ".", n, ".rds")

	if (!file.exists(output_filename)) {
		input <- file(input_filename, open="r")

		message("Generating n-grams; n='", n, "', input='", input_filename, "', output='", output_filename, "'")
		readLines(con=input, encoding="UTF-8", skipNul=T) %>%
		tokenize_ngrams(n=n) %>% unlist(recursive=F) %>%
		# here: each row is an n-gram (as "stem1 stem2 ... stemN" strings)
		tibble::as_tibble() %>% transmute(ngram=value) %>% count(ngram, sort=T) %>%
		saveRDS(file=output_filename)

		close(input)
	}
}

generate_ngrams <- function(input_basename, n) {
	input_filename = paste0(input_basename, ".subs")
	output_filename = paste0(input_basename, ".", n, ".grams")

	if (!file.exists(output_filename)) {
		input <- file(input_filename, open="r")
		output <- file(output_filename, open="w")
		lines_per_block <- 100000

		message("Generating n-grams; n='", n, "', input='", input_filename, "', output='", output_filename, "'")
		total_lines <- 0
		while (T) {
			lines <- readLines(con=input, n=lines_per_block, encoding="UTF-8", skipNul=T)
			num_lines = length(lines)
			if (num_lines <= 0) break

			lines %>%
			tokenize_ngrams(n=n) %>% unlist(recursive=F) %>%
			# here: each row is an n-gram (as "stem1 stem2 ... stemN" strings)
			writeLines(con=output)

			total_lines <- total_lines + num_lines
			message("  Processed block; lines='", total_lines, "'")
		}
		close(output)
		close(input)
	}
}

count_ngrams <- function(input_basename, n) {
	input_filename = paste0(input_basename, ".", n, ".grams")
	output_filename = paste0(input_basename, ".", n, ".csv")

	if (!file.exists(output_filename)) {
		# NOTE: sorry, I'm in no mood to rewrite this in R, handling big files, memory, etc.
		cmd = paste0("sort ", input_filename, " | uniq -c | sort -nr | awk '{$1=sprintf(\"%d, \", $1); print}' > ", output_filename)
		message("Counting n-grams; n='", n, "', input='", input_filename, "', output='", output_filename, "'")
		system(cmd)
	}
}

input_basenames = c("final/en_US/en_US.blogs", "final/en_US/en_US.news", "final/en_US/en_US.twitter")

sample_words <- collect_subsentences("sample") %>% keep_most_frequent(pct=0.9)
#for (bn in input_basenames) {
#	collect_subsentences(bn) # produces xyz.subs
#}


#for (bn in input_basenames) {
	#for (n in c(1,2,3,4)) {
	#	generate_ngrams(bn, n) # produces xyz.n.grams
	#	count_ngrams(bn, n) # produces xyz.n.csv
	#}
#}

# cat *subs | tr ' ' '\n' | sort | uniq -c | sort -nr >words.list

qwer <- function(n, n_min) {
	subsentences %>%
		tokenize_ngrams(n=n, n_min=n_min) %>% unlist(recursive=F) %>%
		# here: each row is an n-gram (as "stem1 stem2 ... stemN" strings)
		tibble::as_tibble() %>% transmute(ngram=value) %>% count(ngram, sort=T)
}

# x <- read.csv(filename, header=F, col.names=c("n", "ngram"), colClasses=c("numeric", "character"))
