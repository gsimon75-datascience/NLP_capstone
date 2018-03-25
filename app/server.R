library(shiny)
library(stringr)

# Helper for enabling/disabling ActionButtons
setActionButtonDisabled <- function(session, id, state) {
	session$sendCustomMessage(type="jsCode",
		list(code=paste("$('#", id, "').prop('disabled',", if (state) "true" else "false", ")", sep="")))
}

setFocus <- function(session, id) {
	session$sendCustomMessage(type="jsCode",
		list(code=paste("document.getElementById(\"", id, "\").focus()", sep="")))
}

## Global constants
N_sug <- 4

# 0=off, 1=critical, 2=error, 3=warning, 4=info, 5=debug, 6=verbose
debug <- 5

## Read the list of known words
words <- readRDS("word_t.rds")

## Create a lookup table (from word to id)
word_to_id <- as.list(words$id)
names(word_to_id) <- words$word
# > word_to_id[["beer"]]
# [1] 21118

## Create a lookup table (from id to word)
id_to_word <- as.list(words$word)
names(id_to_word) <- words$id
# > id_to_word[[toString(21118)]]
# [1] "beer"

## Calculate (unconditional) odds of each word
total_words <- sum(words$occurences)
word_odds <- as.list(words$occurences / (total_words - words$occurences))
names(word_odds) <- words$id

## Create a word list sorted in decreasing order of factors
x <- sort(unlist(word_odds), decreasing=T)
words_sorted <- sapply(attr(x, 'name'), function(x) { id_to_word[[x]] })
names(words_sorted) <- NULL

## Read the Bayesian relatives table
bayes <- readRDS("bayes_t.rds")

## Read the N-gram prefixes table
prefixes <- readRDS("prefix_t.rds")

## Read the N-grams table
ngrams <- readRDS("ngram_t.rds")

shinyServer(
	function(input, output, session) {
		interactive <- reactive({
			input$interactive
		})

		# The 'Interactive' checkbox toggles the 'Generate' button between enabled and disabled
		observeEvent(input$interactive, {
			setActionButtonDisabled(session, "generate", interactive())
		})

		# The 'Generate' button generates a prediction set and displays it on the 'Suggestion #N' buttons
		observeEvent(input$generate, {
			g <- guess()
			display_guess(g)
		})

		# In 'Interactive' mode the 'UserInput' field generates a prediction set and displays it
		observeEvent({ if (input$interactive) input$userinput else 0}, {
			if (input$interactive) {
				g <- guess()
				display_guess(g)
			}
		})

		# The 'Suggestion #N' buttons generate a prediction set and append the chosen one to the 'UserInput'
		observeEvent(input$guess1, {
			g <- isolate(guess())
			append_to_input(g[1])
		})

		observeEvent(input$guess2, {
			g <- isolate(guess())
			append_to_input(g[2])
		})

		observeEvent(input$guess3, {
			g <- isolate(guess())
			append_to_input(g[3])
		})

		observeEvent(input$guess4, {
			g <- isolate(guess())
			append_to_input(g[4])
		})

		append_to_input <- function(word) {
			sentence <- isolate(input$userinput)
			if (isolate(input$interactive)) {
				# remove the already entered partial hint
				sentence <- sub("[^[:space:]]*$", "", sentence)
			}
			sentence <- trimws(sentence)
			sentence <- if (sentence == "") word else paste(sentence, word)
			updateTextInput(session, "userinput", value=paste(sentence, ""))
			setFocus(session, "userinput")
		}

		# Display the predictions
		display_guess <- function(g) {
			update_button("guess1", g[1])
			update_button("guess2", g[2])
			update_button("guess3", g[3])
			update_button("guess4", g[4])
		}

		update_button <- function(btn, value) {
			if (is.na(value)) {
				updateActionButton(session, btn, label="?")
				setActionButtonDisabled(session, btn, T)
			} else {
				updateActionButton(session, btn, label=value)
				setActionButtonDisabled(session, btn, F)
			}
		}

		# Generate a prediction set
		guess <- reactive({
			# the prediction input depends on whether we are in interactive mode or not
			if (interactive()) {
				# interactive mode: treat last word as unfinished -> it is a hint
				x <- str_match(input$userinput, "(.*?)([^[:space:]]*)$")[1,]
				prefixline <- x[2]
				hint <- x[3]
			} else {
				# non-interactive mode: treat last word as finished -> part of the search prefix
				prefixline <- input$userinput
				hint <- ""
			}

			prefix_words <- strsplit(normalise_line(prefixline), "[[:space:]]+")[[1]]
			prefix_ids <- sapply(unlist(lapply(prefix_words, function(w) {word_to_id[[w]]}), use.names=F), toString)
			# NOTE: The NULLs that were produced by unrecognised words are implicitely dropped

			generate_prediction(prefix_ids, hint)
		})

		# Normalise a line (canonicalise punctuation, remove invalid words, fix some typos, etc.)
		normalise_line <- function(line) {
			line %>%
			tolower() %>%
		    # revert unicode punctuation to ascii
		    gsub(pattern="[\u2018-\u201b\u2032\u2035`]",       replacement="'") %>%
		    gsub(pattern="[\xab\xbb]",                         replacement="\"") %>%
		    gsub(pattern="[\u201c-\u201f\u2033\u2034\u2036\u2037\u2039\u203a\u2057]", replacement="\"") %>%
		    gsub(pattern="[\xad]",                             replacement="-") %>%
		    gsub(pattern="[\u2010-\u2015\u2043]",              replacement="-") %>%
		    gsub(pattern="[\u2024\u2027]",                     replacement=".") %>%
		    gsub(pattern="\u2025",                             replacement="..") %>%
		    gsub(pattern="\u2026",                             replacement="...") %>%
		    gsub(pattern="[\u2000-\u200d\u2060\u202f\u205f]+", replacement=" ") %>%
		    gsub(pattern="\u2063",                             replacement=",") %>%
		    gsub(pattern="\u2052",                             replacement="%") %>%
		    gsub(pattern="[\u204e\u2055\u2062]",               replacement="*") %>%
		    gsub(pattern="\u2052",                             replacement="%") %>%
		    gsub(pattern="\u2064",                             replacement="+") %>%
		    # no crlf, bom, lrm, rlm, etc.
		    gsub(pattern="[\r\x80-\xbf\xd7\xf7]",              replacement="") %>%
		    gsub(pattern="[\ufeff\u200e\u200f]",               replacement="") %>%
		    # quotes, parentheses, underscores to space
		    gsub(pattern="[][@{}<>()\"\\|~`:*/%_#$,;+=^0-9-]", replacement=" ") %>%
		    gsub(pattern="&\\S*",                              replacement=" ") %>%
		    gsub(pattern="^|$",                                replacement=" ") %>%
			# zap non-alpha in front of words
			gsub(pattern=" [^a-zA-Z]+",                        replacement=" ") %>%
			# zap non-alpha at the end of words
			gsub(pattern="[^a-zA-Z]+ ",                        replacement=" ") %>%
			# remove those without at least a letter
			gsub(pattern=" [^a-zA-Z]+ ",                       replacement=" ") %>%
			# some common typos (NOTE: standalone only)
			gsub(pattern=" aint ",         ignore.case=T,      replacement=" ain't ") %>%
			gsub(pattern=" cant ",         ignore.case=T,      replacement=" can't ") %>%
			gsub(pattern=" couldnt ",      ignore.case=T,      replacement=" couldn't ") %>%
			gsub(pattern=" doesnt ",       ignore.case=T,      replacement=" doesn't ") %>%
			gsub(pattern=" dont ",         ignore.case=T,      replacement=" don't ") %>%
			gsub(pattern=" didnt ",        ignore.case=T,      replacement=" didn't ") %>%
			gsub(pattern=" hasnt ",        ignore.case=T,      replacement=" hasn't ") %>%
			gsub(pattern=" havent ",       ignore.case=T,      replacement=" haven't ") %>%
			gsub(pattern=" hadnt ",        ignore.case=T,      replacement=" hadn't ") %>%
			gsub(pattern=" isnt ",         ignore.case=T,      replacement=" isn't ") %>%
			gsub(pattern=" arent ",        ignore.case=T,      replacement=" aren't ") %>%
			gsub(pattern=" wasnt ",        ignore.case=T,      replacement=" wasn't ") %>%
			gsub(pattern=" werent ",       ignore.case=T,      replacement=" weren't ") %>%
			gsub(pattern=" wont ",         ignore.case=T,      replacement=" won't ") %>%
			gsub(pattern=" wouldnt ",      ignore.case=T,      replacement=" wouldn't ") %>%
			gsub(pattern=" mustnt ",       ignore.case=T,      replacement=" mustn't ") %>%
			gsub(pattern=" neednt ",       ignore.case=T,      replacement=" needn't ") %>%
			gsub(pattern=" oughtnt ",      ignore.case=T,      replacement=" oughtn't ") %>%
			gsub(pattern=" shant ",        ignore.case=T,      replacement=" shan't ") %>%
			gsub(pattern=" shouldnt ",     ignore.case=T,      replacement=" shouldn't ") %>%
			gsub(pattern=" id ",           ignore.case=T,      replacement=" I'd ") %>%
			gsub(pattern=" youd ",         ignore.case=T,      replacement=" you'd ") %>%
			gsub(pattern=" hed ",          ignore.case=T,      replacement=" he'd ") %>%
			gsub(pattern=" youd ",         ignore.case=T,      replacement=" you'd ") %>%
			gsub(pattern=" theyd ",        ignore.case=T,      replacement=" they'd ") %>%
			gsub(pattern=" ive ",          ignore.case=T,      replacement=" I've ") %>%
			gsub(pattern=" youve ",        ignore.case=T,      replacement=" you've ") %>%
			gsub(pattern=" weve ",         ignore.case=T,      replacement=" we've ") %>%
			gsub(pattern=" theyve ",       ignore.case=T,      replacement=" they've ") %>%
			gsub(pattern=" im ",           ignore.case=T,      replacement=" I'm ") %>%
			gsub(pattern=" youre ",        ignore.case=T,      replacement=" you're ") %>%
			gsub(pattern=" hes ",          ignore.case=T,      replacement=" he's ") %>%
			gsub(pattern=" shes ",         ignore.case=T,      replacement=" she's ") %>%
			gsub(pattern=" theyre ",       ignore.case=T,      replacement=" they're ") %>%
			gsub(pattern=" youll ",        ignore.case=T,      replacement=" you'll ") %>%
			gsub(pattern=" itll ",         ignore.case=T,      replacement=" it'll ") %>%
			gsub(pattern=" theyll ",       ignore.case=T,      replacement=" they'll ") %>%
			gsub(pattern=" couldve ",      ignore.case=T,      replacement=" could've ") %>%
			gsub(pattern=" shouldve ",     ignore.case=T,      replacement=" should've ") %>%
			gsub(pattern=" wouldve ",      ignore.case=T,      replacement=" would've ") %>%
			gsub(pattern=" lets ",         ignore.case=T,      replacement=" let's ") %>%
			gsub(pattern=" thats ",        ignore.case=T,      replacement=" that's ") %>%
			gsub(pattern=" heres ",        ignore.case=T,      replacement=" here's ") %>%
			gsub(pattern=" theres ",       ignore.case=T,      replacement=" there's ") %>%
			gsub(pattern=" whats ",        ignore.case=T,      replacement=" what's ") %>%
			gsub(pattern=" whos ",         ignore.case=T,      replacement=" who's ") %>%
			gsub(pattern=" wheres ",       ignore.case=T,      replacement=" where's ") %>%
			gsub(pattern=" noones ",       ignore.case=T,      replacement=" noone's ") %>%
			gsub(pattern=" everyones ",    ignore.case=T,      replacement=" everyone's ") %>%
			gsub(pattern=" nowheres ",     ignore.case=T,      replacement=" nowhere's ") %>%
			gsub(pattern=" everywheres ",  ignore.case=T,      replacement=" everywhere's ") %>%
			gsub(pattern=" yall ",         ignore.case=T,      replacement=" y'all ") %>%
			gsub(pattern=" bday ",         ignore.case=T,      replacement=" birthday ") %>%
			gsub(pattern=" dis ",          ignore.case=T,      replacement=" this ") %>%
			gsub(pattern=" dat ",          ignore.case=T,      replacement=" that ") %>%
			gsub(pattern=" i ",                                replacement=" I ") %>%
			gsub(pattern=" i'",                                replacement=" I'") %>%
			# some abbreviations (NOTE: standalone only)
			gsub(pattern=" c'mon ",        ignore.case=T,      replacement=" come on ") %>%
			gsub(pattern=" (be?)?c[ou]z ", ignore.case=T,      replacement=" because ") %>%
			gsub(pattern=" imma ",         ignore.case=T,      replacement=" I'm a ") %>%
			gsub(pattern=" ofcoz ",        ignore.case=T,      replacement=" of course ") %>%
			gsub(pattern=" pl[sz] ",       ignore.case=T,      replacement=" please ") %>%
			gsub(pattern=" ppl ",          ignore.case=T,      replacement=" people ") %>%
			gsub(pattern=" tho ",          ignore.case=T,      replacement=" though ") %>%
			gsub(pattern=" u ",            ignore.case=T,      replacement=" you ") %>%
			gsub(pattern=" ur ",           ignore.case=T,      replacement=" your ") %>%
			# zap all single characters except 'I' and 'a'
			gsub(pattern=" [^IAa] ",                           replacement=" ") %>%
			# everything where a letter repeats more than 2 times
			gsub(pattern=" [^[:space:]]*([^[:space:]])\\1{3,}[^[:space:]]* ", ignore.case=T, replacement=" ") %>%
			# coalesce whitespaces
			gsub(pattern="\\s+",                               replacement=" ") %>%
			trimws()
		}


		# Generate predictions for a given prefix and hint
		generate_prediction <- function(prefix_ids, hint) {
			if (debug >= 4) print(paste0(Sys.time(), " generate_prediction([", paste(prefix_ids, collapse=", "), "], '", hint, "')"))

			# First, iterate though the prefixes and collect what other words does the Bayes table suggest ('conditional'), provided that this particular prefix is present ('condition') in the sentence
			if (debug >= 5) print(paste0(Sys.time(), " Searching Bayesian relative words;"))
			bayes_factors <- list()
			for (pfx in prefix_ids) {
				relatives <- bayes[bayes$condition == pfx, c("conditional", "factor")]
				if (debug >= 6) print(paste0(Sys.time(), " Found Bayesian relatives; word=", pfx, ", num=", nrow(relatives)))
				for (idx in 1:nrow(relatives)) {
					f <- relatives$factor[idx]
					if (is.na(f))
						next
					i <- toString(relatives$conditional[idx])
					# Skip unsuitable items
					if (debug >= 6) print(paste0(Sys.time(), "   relative ", idx, "; id=", i, ", factor=", f))
					if (!startsWith(id_to_word[[i]], hint))
						next
					# Chain this factor to the already collected ones
					if (is.null(bayes_factors[[i]])) {
						# We start with the unconditional odds of that suggested word 'i'
						bayes_factors[i] <- word_odds[[i]] * f
					} else {
						# Then chain all the odds-multiplying factors
						bayes_factors[i] <- bayes_factors[[i]] * f
					}
				}
			}

			# Then find the longest known prefix that matches the end of what we got
			if (debug >= 5) print(paste0(Sys.time(), " Searching for longest known N-gram prefix;"))
			prefix_id <- -1
			for (i in length(prefix_ids):1) {
				next_id <- prefixes[(prefixes$parent == prefix_id) & (prefixes$word == prefix_ids[i]), "id"]
				if (length(next_id) == 0)
					break
				prefix_id = next_id
			}
			if (debug >= 5) print(paste0(Sys.time(), " Found prefix; id=", prefix_id))

			result <- vector()
			more_needed <- N_sug

			# Find the N-grams that start with the found prefix
			if (debug >= 5) print(paste0(Sys.time(), " Searching for N-gram followers;"))
			ngram_m <- list()
			followers <- ngrams[ngrams$prefix == prefix_id, c("follower", "occurences", "factor")]
			num_followers <- nrow(followers)

			if (num_followers > 0) {
				if (debug >= 5) print(paste0(Sys.time(), " Found N-grams followers; n=", num_followers))
				for (idx in 1:num_followers) {
					i <- toString(followers$follower[idx])
					f <- followers$factor[idx]
					if (is.na(f))
						f <- 1e100
					# Skip unsuitable items
					if (debug >= 6) print(paste0(Sys.time(), "  follower ", idx, "; id=", i, ", factor=", f))
					if (!startsWith(id_to_word[[i]], hint))
						next
					# Combine the Bayesian factors into the ngram-based ones
					if (i %in% bayes_factors) {
						f <- f * bayes_factors[[i]]
						bayes_factors[[i]] <- NULL # remove to prevent double suggestions later
					}

					# Calculate an expected value (remember: f is an odds)
					ngram_m[i] <- followers$occurences[idx] * f / (1 + f)
				}

				# Sort the N-gram hints and 
				ngram_hints <- sort(unlist(ngram_m), decreasing=T)
				ngram_hints <- sapply(attr(ngram_hints, 'name'), function(x) { id_to_word[[x]] })
				names(ngram_hints) <- NULL
				result <- append(result, ngram_hints[1:min(more_needed, length(ngram_hints))])
				more_needed <- N_sug - length(result)
			}

			# If there are too few results, top it up from the Bayesian relatives' list
			if ((more_needed > 0) && (length(bayes_factors) > 0)) {
				if (debug >= 5) print(paste0(Sys.time(), " Topping up with Bayesian relatives; needed=", more_needed))
				bayes_hints <- sort(unlist(bayes_factors), decreasing=T)
				bayes_hints <- sapply(attr(bayes_hints, 'name'), function(x) { id_to_word[[x]] })
				names(bayes_hints) <- NULL
				result <- append(result, bayes_hints[1:min(more_needed, length(bayes_hints))])
				more_needed <- N_sug - length(result)
			}

			# If there are still too few results, top it up from the (unconditional) global word list
			if (more_needed > 0) {
				if (debug >= 5) print(paste0(Sys.time(), " Topping up with dictionary words; needed=", more_needed))
				for (global_hint in words_sorted) {
					if (!startsWith(global_hint, hint)) next
					if (global_hint %in% result) next
					result <- append(result, global_hint)
					if (length(result) >= N_sug) break
				}
				more_needed <- N_sug - length(result)
			}

			if (debug >= 4) print(paste0(Sys.time(), " Result: [", paste0(result, collapse=", "), "]"))
			result
		}

	}
)

