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


## Implementation of the Demo App

It's quite simple and straightforward, though it had some bumps and hiccups as well.

Giving predictions for a given sentence and an optional hint (starting letters already typed by the user) goes like this:

1. Normalise the sentence exactly as the training process did with the corpus

2. Tokenise it: convert the sentence to a sequence of word IDs

3. Collect the Bayesian classifiers for all the words of the sentence.
As a result we get a classifier for *all other* words, that describe how much their appearance is affected by
the known part of the sentence.

3. Identify the longest known prefix that matches the end of the sentence

4. Get the candidate followers that may follow this prefix, along with their classifiers

5. Fix these N-gram follower classifiers up by the ones we got for them in Step 3.
In the same time, remove the used-up classifiers from the Step3-list.

6. Convert these word-classifiers to real odds by multiplying with the absolute odds of the words.

7. Sort them by these odds and add to the results' list

8. If we need more predictions, then use the rest of the list from Step3 to produce them.
These words won't be related to the end of the sentence, but to the sentence as a whole, but it's still
better than a wild guess.

9. If we still don't have enough predictions (may happen for short sentences of rare words), then
just take the absolute frequency of words, and fill the remaining prediction slots by them (avoiding
the duplicates, of course).

10. If there is a hint (starting letters already typed), then all the 'add to results list' has a
filter that lets through only those words that match the hint.


I needed some JavaScript trickery to give a more intuitive UI behaviour:

* When the user chooses a non-interactive mode, the 'Generate' button must be disabled
* The predictions must appear on ActionButtons, so they are clickable
* When we have too few predictions (in interactive mode, the hint matches too few known words),
then those buttons must be disabled as well.
* After clicking a prediction result button, the focus must be returned to the Input textbox
* While the calculations are in progress, a spinner must be displayed

Otherwise the UI code is quite straightforward and quasi self-documenting :D, though I had some new
experience with R as well.

### Hashmaps / associative arrays in R

For example, to convert words to IDs.

* Either you store them in a dataframe, one column for name, one for ID, and pay the linear lookup
to find the row.
* Or create a list, where the names are the words and the items are the IDs. Apart from how many times
did I mistype the `yadda[[idx]]` notation and how it produced silent but unwanted results, it seemed to
work. But it's still not effective, I guess it's linear too.
* Or you use environments, another tool that was designed for something else.
Now this one is indeed O(log(N)), but it's a bit counter-intuitive.

Environments also use the double-bracket notation, but for a nonexistent item they return NULL, and the getter of the keys of the hashmap is called `ls()`. No real problem here, only I wouldn't guess it without stackoverflow :D.


### `apply` and friends

The `apply` family is one of Rs real strengths, a real prize thing, use them if you can, they are blazingly faster than manual looping, `mapply` is my favourite one.

Strange, but there's no 'execute only' in the base package, only in `plyr::l_ply`.
(I could've just discarded the result of `sapply`, but we're talking about performance, so producing something just to be discarded is a waste of CPU time.)


NOTE: `Vectorize` is just syntactic sugar around the `apply` family, it won't really make a one-item-only custom function faster than executing it in a loop.


### Awkward solutions

Sorting some index-value pairs (like the Bayesian classifiers) by value is a pain. We can sort a list or vector of values, but that's only the values.

To incorporate the index, we must make it a list, store the index as 'name' of the values, and later retrieve it, and converted from text to numeric again.

Or make it a dataframe, maybe.

I badly miss the usual lower-level data structures in R, like hashmap (from any hashable type to any type), or object references, and I'd gladly trade the 'lists' for such.

I know, R is designed to do vectorised operations on column-oriented data, and it's very effective in such, but then this kind of task would require another kind of language.

Shortly expressed: Sometimes it felt like badly misusing R, but I found no nicer way to do it.



## Implementation of the Training Process

Now, this was quite a long journey, but I gained lot of experience by it, and I'm glad that
I did it in this training assignment, and not on a real-life job :D.


### The model training process

1. We normalise the corpus: canonicalise the punctuations, remove unwanted characters,
fix some common typos, resolve some abbreviations, and just discard the junk
(eg. words in which a letter repeates more than twice, so no 'coool' and 'cooool' and 'coooooool', sorry)

2. Count the words, combining those that differ only by capitalisation. (Less caps wins, so
NO SHOUTING WILL REMAIN, if those words occur in lowercase as well.)

3. Discard all single words (now), or up to a predefined criteria (planned).

4. Assign IDs to these words and store them into the DB. At this point around 290k words remained.
Considering that it contains all inflections (4 for nouns, 5 for verbs, 1 for the rest) and a lot of garbage as well, it's realistic.

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

12. DB cleanup: Remove some columns (eg. some occurence counters) and indices that were needed only for the post-processing.
Also DB defragmenting, to free up the deleted blocks and rearrange the rest for faster access.

All my alpha tests were in Python (see the end of `build_model.py`), including tests for prediction speed,
so DB access was worth it to have a clean picture.

13. Exporting data to R: from SQL tables to RDS files, and also reassigning of word IDs.

In the last step the SQL tables are read into R objects and saved as RDS files. A positive surprise here is that while
the SQL files are uncompressed (because they are read/write data stores), but RDS files are highly compressed, so the
~600 MB SQL database just collapsed to RDS files in a total of 40 MB-s.

Upon creation, the words get nice sequential IDs, but in a post-processing step some (most...) of these words are discarded, so these IDs will not remain
contiguous.

When reading the words' properties in an R dataframe, it would be handy to just access a word by its row number, so we needed a Word-ID-to-row-number translation, or more preferably, the word IDs be equivalent to row numbers.

Word IDs are referred by Bayesian pairs, prefixes and N-gram followers alike, so re-assigning them is a pain, but it has to be done once in the model preparation phase, and then in the runtime app it makes everything a lot easier.



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

To access both computing power and memory, I utilised a virtual machine in *Google Cloud Compute Services*,
so assigning more RAM or CPUs were just a matter of clicking on a web UI (and a few bucks per day when I
needed some unusual amount), so I had quite more options compared to working on a personal laptop.

So, Bayes pair generation still worked in the python implementation (it's still there in `build_model.py`,
although it took about 3 hours to complete. At this point I was already content with having the result in
the same day...

#### Collecting the N-grams

First I tried just collecting the N-grams in a big dict, ran out of memory within the hour.
Then experimented a bit with dicts and learned that it uses about 72 bytes per stored object, so at least I knew the reason.

For a quick try, I reorganised the thing to use lists, memory seemed ok, linear searching scaled badly, as expected.

As the usage consists of few inserts and tons of queries, keeping lookup O(log(n)) is a must, even for the cost of
insertions being O(n), so added some lines to keep the lists sorted and do the lookups by bisect, that helped.

At this point I planned to keep only parts of the data in memory (as much as fits), and when it's full, then
commit the counters to the DB adding them to the ones from previous runs.

This required storing the IDs along with the N-grams, and knowing how much memory is available (see a paragraph on this topic later).

Needless to say, ran out of memory and started cache-thrashing, because:
* Storing the IDs means adding 20-100% to memory consumption, because our entities are otherwise small
* We need the double of this, because when committing to DB, all data is available once in python (as it is pushing it) and
once in the DB (as it caches it and pushes on to disk).
* pythons memory manager can free only its objects, but not its integers, so when it ate up the RAM,
there wasn't enough available for the DB to work effectively.

Soon I got fed up with fighting with Linuxes memory management (see the topic on memory management), junked all the caching-
flushing logic and decided to store everything in RAM, and commit only at the end.

Thanks to Google Cloud Compute, I was able to throw a 52 GB (!) of RAM to that node, so collecting the N-grams ran in 3-4 hours,
and at its peak it used 40 GB of the RAM. (For 4 hours it costed me about 80 cents, I say it was worth it, considering the day
or so I spent on twiddling with caching-flushing logics.)

Then I realised that I still must separate the last word (the one to be guessed) from the prefix (the ones at the end of the user input),
and then merge those prefixes got me again, it could be estimated to complete in 2-3 days, even doing it completely in RAM.


A little sidetrack, I'm meaning rearranging these N-grams:

* C-B-A - P =  8  occurences
* C-B-A - Q = 59  occurences
* D-B-A - P =  3  occurences
* C-D-A - P = 17  occurences
* C-D-A - Q =  5  occurences

Into something like this:

Prefixes:
* C-B-A = (1) = 67 occurences
* D-B-A = (2) =  3 occurences
* C-D-A = (3) = 22 occurences

Full N-grams:
* (1)-P =  8
* (1)-Q = 59
* (2)-P =  3
* (3)-P = 17
* (3)-Q =  5

It was estimated to take days, mostly due to flinging all the otherwise duplicated data around.

So the next idea was to collect the data as it will be needed at the end (obviously...)

That means a tree of prefixes, because they must be evaluated from the last word, expanding towards the beginning
of the sentence, and if we want to collect all the N-grams up to N, then at each node we must stitch a collection of followers:

```
A ........... { P: 28, Q: 64 }
|
+-B ..........{ P: 11, Q: 59 }
| |
| +-C ........{ P:  8, Q: 59 }
| |
| +-D ........{ P:  3 }
|
+-D ..........{ P: 17, Q:  5 }
  |
  +-C ........{ P: 17, Q:  5 }
```

Structures got bigger, insertions required more stuff to move around, so it's no big surprise.

I still had around 3 days till the deadline and I had only some minimalistic skeleton of the actual
prediction logic, the client-side, and realising that I'm again struggling with the tools and not with
the problem, I decided to switched again, for the last time.

I took a quick tour with some Google technogies:
* BigQuery is a more-or-less SQL database for *huge* amounts of data, it's billed on data access,
and unfortunately knows only full-table-scans, so it'd be both slow and costly.
* DataPrep is an exploratory analysis tool, not for this purpose
* DataProc uses Apache Spark, seemed related
* DataFlow uses Apache Beam, also seemed related

DataFlow even has a tutorial example that counts words in a corpus (really!),
so I spent a few hours exploring it.

#### Google DataFlow

We may create data processing pipelines (coding in python or in Java), the pipelines may fork and join,
forming any kind of Directed Acyclic Graph, performing any transformation that we can code.

These pipelines can be fed from files either local or on Google Cloud Storage, from databases, and the
output can also be sent to any of these.

They can be executed locally (for debugging), or on a farm of worker nodes that are created and managed
automatically on Google Compute Engine.

This sounded very promising, at least for generating the N-grams, but unfortunately the counting of the
N-grams is different.

If we split this process to several nodes, then the nodes may not cache anything locally, so the 
468M N-gram instances would mean 468M DB operations, all bombarding one single DB engine.

This concept would indeed work, and scale as much as possible, though it would inevitably incur a
significant communication overhead.

*NOTE*: If I could't have managed to solve the issue in the RAM on a single machine, I would've
chosen this approach! (And I'm going to play this scenario anyway, just for the fun of it :D ...)


#### Back to the single-node approach

All the slowness and memory consumption was caused because overhead of the memory management of
the high-level languages like R and python, and because their data structures are generic by nature, and
cannot be tailored to the purpose.

None of these restrictions apply to lower-level languages like the good old C.

NOTE: I used some C++ features here for making the code easier to maintain, but the memory management
is fully plain-C, C++ is just used as syntactic sugar here.

As the pointers to the followers' lists and the childrens list are stored directly in the prefixes' structures,
we don't need prefix IDs, nor N-gram IDs, that makes it 4 bytes less per instance. (Don't laugh, we're talking
about tens of millions of instances...)

Each such structure is so small that it is comparable with the memory chunk allocation overhead, so it
is not worth allocating them separately and maintaining only pointers to them, but we may just
store them contiguously and move when inserting a new item. (It's slower by a factor about 5, but consumes
about half the memory.)

So, the resulting memory structure is primitive as an axe, and it's a real pain when inserting new records,
but for locating an existing entry it is fast, and I mean *very fast* here, and it has the least possible
memory overhead.

The results of it? Collecting the Bayesian pairs and N-grams in total, it required less than 9 GB RAM,
and ran in less than *an hour*. (Yes, nine gigabytes and one hour. Compare this to above 40 GB and just short
of a week.)

The moral of the story here is that if you don't mind that the code is one-purpose and unmaintainable,
and local to one machine, and that you must code literally everything by yourself, then nothing beats the
good old custom-tailored optimised C.

(Maybe the good old custom-tailored optimised Assembly, but I had no time for that, and C was good enoug for now :D.)


### That note (rant...) on memory management


#### Overcommitting

Linux has a dangerous policy: it promises all processes all the memory they ask, and when they actually try
to use it, and the memory runs dry, the kernel just shoots some processes in the head to get rid of the observers.

It's called 'memory overcommitting', and it's an ugly hack, loved by lousy programmers who allocate more
that they use, and hated by everyone who actually wants to use what he allocated.

To turn it off, you should `echo 2 > /proc/sys/vm/overcommit_memory`, but it will make a Google Compute
node unusable, because its memory sharing relies on the overcommit mechanism :(.

Here you can say `echo 98 >/proc/sys/vm/overcommit_ratio`, and so instructing the kernel to let allocate 98%
of the total memory (that 2% is usually enough for the kernel itself).

One bomb defused, my app will no longer end with an 'Process killed by OOM' message after days of running...


#### Manual caching

I wanted to cache N-gram counters while I have memory, and then flush them to DB. So, how much memory
do I have, when has it ran out?

Don't wait for pythons 'MemoryError' exception, it will come, but not where you expect. Maybe your
last cache allocation will still leave some bytes free, and it'll be the next local variable or function
argument that cannot be allocated, and there you go.

So I wanted to ask how much is available, and stop chowing it up before say 100M short of the end.

Hundred writings on the net recommended hundred methods, all based on assumptions and guesses.

Except for [this one](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=34e431b0ae398fc54ea69ff85ec700722c9da773)

The library that uses it is `psutil`, so it's available, but note that it involves opening
the pseudo-file '/proc/meminfo', reading textual data from it, and parsing the numbers from that,
so it's not that cheap that you want to do it a million times in a second.

It's not the shame of `psutil`, the value is just not exported from the kernel in any other
way than through `/proc/meminfo`, at least I found no other in the sources...

