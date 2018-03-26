# Predictive Text Input

There is a short [presentation](http://rpubs.com/gsimon75/PredictiveTextInput) about this project,
and you can also [try it](https://gsimon75.shinyapps.io/PredictiveTextInput/).

This lengthy text aims to collect the things I learned during this assignment (mostly for the future me),
so it's more like a diary than a catalogue.

If you're just interested in the actual code, go ahead and check the files, but be aware that I 
started to run out of time, so the codebase is somewhat chaotic, meaning that 
it's not organised prettily, and at some places even the remnants of failed attempts haven't
been removed.

Start with `build_model.py`, proceed with `find_ngrams.cc` and finish with `cleanup.sql`, these are
the main 3 files, the rest is mostly scratchpads, API tests, and so on.


## Concepts

The idea is that quite much of the words we type is actually predictable.

Not the ones that carry genuine information, so if the user types "My favourite day of the week is ...",
then no science can extract information of the sentence, because it is just not there.

We may know that there are 7 possible outcomes (if we can parse the grammar, identify the Parts-of-Speech,
etc.), and may even know something about the distribution of them (very few people like Mondays :)),
but still a lot of guessing would be involved.


### Positional dependence

On the other `_____`, we quite well recognise that here it was the word `hand` missing, because that 4 words
are not independent, they form an idiomatic expression.

If we maintain a dictionary of these, along with the occurence counts, and we recognise some such word-pairs
or triplets or quadruplets, then we may have an idea about what might come next.

### Tematic dependence

If we read the word 'bill' in a sentence, we usually know whether it refers to a polearm, to a banknote,
to a list of purchased goods or to some legal document.

We know it even without actually parsing the sentence, even from an incomplete fragment, but then it is noticeable,
that the shorter the fragment, the greater the confusion.

Strange as it is, if we take some arbitrary permutation of the words of that sentence, our chances to
determine the actual meaning of that word are no less than in the normal order.

If there are other 'legalese' expressions around it, then it'll refer to the document, if the other words
are consumption-related, then it'll be the pay-list, etc.

So there is some correlation between the words in the same sentence, regardless of their position.


### N-grams

The positionally grouped words can be identified if we simply count the occurences of every
different word, then every different pair, and so on to higher N-grams.

At first one might expect that the number of such different N-grams grow exponentially with N,
but surprisingly this is not the case, because not *every* combination occurs, but only
the ones that carry some meaning. Actually there are quite a few such 6-7 long N-grams that
occur repeatedly.

This is good news because we may parse further with a given amount of memory, but it is also bad news,
because az the number of different N-grams is getting less, it also means less information to us.

For the English language the practical limit is around 3-5.

### Word variances

Shall we distinguish 'go', 'goes', 'going' and 'gone'?

If we do, then we split the `<to go> mad` into at least four distinct cases, which is not good,
because together they would be a significant amount, but separately they are quite average.

On the other hand, if we coalesce them, then won't be able to suggest all the different forms,
only one of them.

In English we are quite lucky with so few forms, but for the agglutinative languages (that is,
most of them...) it is a hard problem.


### Bayesian relatives

If a sentence is of some narrower topic, like legal matters, music, science (and sub-topics within),
then a considerable amount of the words will be specific to that topic.

If you find words like 'vitamines', 'cholesterol', 'saturated' and 'carbohydrates', then you may
expect that 'minerals' and 'lipides' will occur above their normal frequency, and 'valve', 'catalyst'
and 'turbocharger' below it.

This information doesn't have too much *direct* value, we can't just suggest 'glucose' at every word position
even if we were 100% sure that it occurs somewhere in the sentence, but its *indirect* value is
obvious:

If, after 'burns away the', our N-grams suggest two equally possible continuations, 'fat' and 'soot',
then it's the presence of other words that can tilt the balance to our benefit.


### N-grams again

Now that we are familiar with the Bayesian classifiers and the concept of 'how much does A affect B',
it's worth applying it to the N-grams as well.

The usual N-gram-based models think like 'Given the last three words C-B-A, what is the probability of
R being the next one?', that is, P(R | CBA).

This doesn't take into account the context of the sentence, that is a chains of classifiers multiplied
together, resultin an overall odds-multipliers for each candidate.

To make use of it, here we also need odds, so I also calculate classifiers for such N-gram prefix + follower
pairs: $\frac{P(R | CBA)}{P(R | \bar{CBA}}$.

That describes *how much is R affected by CBA*, and it can easily be combined with (i.e. multiplied by)
the contextual odds-multipliers.

Basically, we are calculating in odds-domain instead of probability-domain.

Sure, this means that when finally we want the expected probability for each candidate, we must return to
p-domain, but that's just one division per actual candidate.



### Problem of frequent words

When counting what words tend to occur frequently along the condition-word 'valve', we'll find that
the top of the list is 'the', 'in' and 'a' - just like for any other conditions.

So, should we always suggest these 3 words, or what?

It is just half the story how frequently a word appears with 'valve', the other half is how frequently
it appears *without* it.

If it occurs more frequently with it than without, then 'valve' *encourages* that word.

If it occurs more frequently without it than with, then 'valve' *discourages* that word.

If its frequency is about the same, then they are unrelated.

This is what we measure via Bayesian classifiers: the conditional probability of B occuring provided that
A occured, divided by the conditional probability of B occuring provided that A hasn't occured:

	$f = \frac{P(B|A)}{P(B|\bar{A})}$

This quantity acts as an odds-multiplier fixup: if the word B has in general some $\frac{P(B)}{P(\bar{B})}$
odds, then having detected A, this odds will be multiplied by this 'f'.

Interpreting its effect, we basically group those words that 'thematically belong together', without actually defining what do they have in common, just relying on how frequently they occur with and without each other.



### Problem of the rare words

If the word B occured only in one sentence (for example, it's a persons name), then all the other A words
in that sentence will register as 'causing B with 100% certainty', the classifier factors will be infinite.

Consider: B occured once in the many occurences of A, so the nominator is small. But B hasn't ever
occured without A, so the denominator is zero, therefore the classifier is infinite.

In laymans terms: Whenever B happened (once), it *always* happened after A, so if A happens, this is the
only chance for B to occur.

While technically it is true, "if we don't suggest B after A, then we won't suggest it ever", there still
might be far more probable candidates.

It's not fair, but some of the rarest words must be simply discarded.
"It has the right, it just doesn't have the chance."

The proper criterion here would be to discard everything that has no chance ever to make it to the top-N suggestions, but it's easier said than done.

Considering all other possibilities of all N-gram prefixes would be hard but feasible still, but when we involve the Bayesian classifiers from all the other words of the sentence, it gets out of hand.

Now I just discard all single words (they must go anyway), and as a post-filtering as much other low-freq words as needed. (More on this topic later.)



## Implementation

### The model training process

1. We normalise the corpus: canonicalise the punctuations, remove unwanted characters,
fix some common typos, resolve some abbreviations, and just discard the junk
(eg. words in which a letter repeates more than twice, so no 'coool' and 'cooool' and 'coooooool', sorry)

2. Count the words, combining those that differ only by capitalisation. (Less caps wins, so
NO SHOUTING WILL REMAIN, if those words occur in lowercase as well.)

3. Discard all single words (now), or up to a predefined criteria (planned).

4. Assign IDs to these words and store them into the DB. At this point around 290k words remained.
Considering that it contains all inflections (4 for nouns, 5 for verbs, 1 for the rest) and a lot of garbage as well, it's realistic

5. Tokenise the corpus: replace each word with its ID. This gains us a *huge* amount of processing power, as numeric IDs are smaller, require less RAM, and are way faster to handle than free-form strings. And take my word, we do need all advantage we can get...

6. Count the Bayesian relatives. Theoretically this would mean the cross product of the 290k words,
practically the matrix is very sparse, it contains only some 113M elements.

Each containing an ID for the condition, one for the conditional, and one integer for the counter, that's 12 bytes, so it's raw content is around 1.3GB,
don't try to manage it in R. Python deals with it, if you don't use dicts, because they eat up around 70 bytes per item.
(Have I said that I learned *a lot* from this assignment :D ?)

7. Dump the Bayesian counters to the DB.

8. Collect the N-grams and dump them to the DB. This is worth a paragraph on its own.

9. DB post-processing: Calculate classifiers from occurence counters (of word-pairs and of N-gram prefix+follower pairs)

9. DB post-filtering: Discard the words, bayes classifiers and N-grams that occur too rarely.

10. Discard classifiers that are too near to 1.0 (eg. between 1/N and N, for some N)

11. Restore referential integrity: discard all word-pairs and N-grams that refer to discarded words, etc.

12. DB cleanup: Remove columns (eg. some occurence counters) and indices that were needed only for the post-processing.
Also DB defragmenting, to free up the deleted blocks and rearrange the rest for faster access.


### Why python?

R is handy and elegant and *very* efficient at SIMD operations, when we're doing some math on *column-wise*
grouped data, but doing sequential text processing in R is ... well...

Originally I tried to use R as long as I could, and that lasted right up to Step 1., but reading a text file,
applying some 90 regexes to it, split, feed into hashmaps, it's not really what R is for.

It just felt like felling trees with the smartest calculator ever :D ...

Not unlike this, as effective as R is with *tabular* data, as clumsy it is with any other structure, like
hashmaps (lists operate linearly, environments are OK, only you need quirks everywhere), trees, graphs, whatever.

For text processing and structured data manipulating I thought about either perl or python, but expressing deeply
interconnected data in perl tends to get 'write-only' and hard to maintain and debug, so I opted for python.

(N-gram prefixes are tree nodes which refer to their word, their parent, and their follower candidate*s*, which
are (word, counter) pairs. Expressions like this just give the fuel to such slurs like "Perl is the language whose
readability is not affected by RSA encryption." :D )

So python it is, and that's how `build_model.py` gets into the picture.

Despite its name, at its end it also contains the test, because I haven't had the time to
separate the build+test common parts into modules that could've been imported into separate `build.py`
and `test.py` -s. It's was on my list, only there always were more important and urgent issues.


### Why SQLite?

At first I started with internal data structures and saved/loaded them from storage (saveRDS() in R, later cPickle in python),
but whenever the concept changed even a bit (a new field, a different grouping, a different access pattern),
it meant restructuring the data representation, creating and maintaining new lookup tables, not mentioning
that I had to either convert the existing data, or start everything over again.

At this time one such starting-again took around 4-6 hours, not unbearable, but impractical for
early development.

Quite soon I realised that access and representation must be separated, so a DB was needed.

The candidates were SQLite (in-process local DB engine, quite good at it), some server-client
RDBMS like PostgreSQL or MySQL (or MariaDB, or however it's called this week...), or some NoSQL
like MongoDB.

All this data is transient, it's needed only during processing and producing the final .rds
files, no maintenance, scalability and remote access is needed, so most of the benefits of
the server-client DBs were irrelevant here.

On the other hand, for SQLite the usual query communication overhead just doesn't apply,
queries are just in-process function calls, which is fast - as long as it fits on the
local machine.

An important thing: it has APIs to whatever language you want, C, python and R included.


### Why C/C++?

Collecting the N-grams cannot work directly on DB.

If we collect N-grams (length=N), then we are sliding a window on the words of a line,
one by one, so if the line had W words, then we produced W-(N-1) N-grams, that is,
as many as the total words, minus (N-1).

If we have L lines, then it means as many N-grams as the total number of words,
minus L*(N-1).

The corpus contains around 102M words (minus the junk) and 4.2M lines, so it means (106.2 - 4.2*N) million N-grams.

For N=1, N=2, N=3, N=4 and N=5, it means around 468 million N-grams,
so it would mean that many operations.

Counting just 500 us for each step, sliding the window and invoking a
prepared statement, it would still take 65 hours, that is, more than a weekend.

The next step is to 'cache' the counters in memory.

That 468M N-grams comprise of around 20M *kinds* of N-grams, so on average every counter is incremented around 13 times.
If we could keep the counters in RAM, and only commit them to the DB at the end, we'd need only 20M operations.
(On machine level, incrementing a counter is about 0.4 ns, the big deal is to locate that counter in memory.)

# To be continued

