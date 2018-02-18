library(tibble)
library(dplyr)
library(tokenizers)

locale <- "en_US"
sources <- c("blogs", "news", "twitter")

basename <- function(locale, src) {
	paste0("final/", locale, "/", locale, ".", src)
}

collect_subsentences <- function(input_basename) {
	input_filename <- paste0(input_basename, ".txt")
	output_filename <- paste0(input_basename, ".subs")
	words_filename <- paste0(input_basename, ".words.rds")

	all_words <- tibble::as_tibble()
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
			these_words <- lines %>% unlist() %>% tibble::as_tibble() %>% transmute(word=value) %>% group_by(word) %>% summarize(n=n())
			all_words <- all_words %>% rbind(these_words) %>% group_by(word) %>% summarize(n=sum(n))

			# write the output
			lines %>% sapply(function(x) {paste0(x,collapse=" ")}) %>% writeLines(con=output)

			total_lines <- total_lines + num_lines
			message("  Processed block; lines='", total_lines, "'")
		}
		close(output)
		close(input)

		saveRDS(all_words, words_filename)
	} else {
		all_words <- readRDS(words_filename)
	}
	all_words
}

sample_words <- collect_subsentences("sample")


