Predictive Text Input
========================================================
author: Gabor Simon
date: 2018-03-25
autosize: true


<style>
.small-code pre code {
font-size: 1em;
}
</style>
<footer>https://github.com/gsimon75/NLP_capstone</footer>

The idea
========================================================

Typing letters one by `___` takes a lot of time, so let's speed things `__` a bit! As you see, quite a lot `__` our words can be figured `___` from the context, so an intelligent Input Method could do that for us `__` well.

The top 3 categories that are easy to guess:
- Idiomatic expressions ('side to side', 'back and forth')
- Sentence context ('guitar' and 'drums' imply other instruments)
- Absolute word frequency ('the', 'as', 'in', 'a')

The algorithm
========================================================

The secondary guess uses a simple coincidence statistics: which word pairs occur frequently in the same sentence (regardless of position).

The first strategy is based on an N-gram distribution: by the last N words we predict what words might follow them with what chances.

If we still don't know enough to make a guess, fall back to suggesting the most common words.

To avoid the frequent but general words our metrics aren't just probabilities, but Bayesian classifiers $\frac{P(B|A)}{P(B|\bar{A})}$, so we measure *how much a word is affecting the other*, be it positive or negative.


The App
========================================================

Two main parts: suggested words (4) and the input field.

Two operating modes:

#### Interactive
The user may type into the input field, and the predictions are re-calculated after each letter. Choosing a prediction completes the partially entered word, and a new word is started after a space or a punctuation.

#### Non-interactive
The input field can be edited or pasted into, but the predicting must be started manually by a Generate button.
(Mostly useful for entering test sentences.)



Metrics / Details
========================================================

This amount of data required quite a toolkit: the text normalisation was done in Python, the intermediate storage is an SQLite DB, the mass object counting is written in C (for performance), and the demonstration app is written in R+Shiny.

- Source text corpus: ~0.5 GB
- Unfiltered intermediate database: ~26 GB (290k words, 20.6M N-grams, up to N=5)
- Processing requirements: ~8.4 GB RAM, 4 hours
- Filtered database: ~1.3 GB (1k words, 2.3M N-grams)
- Final R RDS files (compressed): ~55 MBytes
- Full-word (no first letters) success rate: ~18-20%
- Efficiency (entered characters / required keystrokes): ~2.1
