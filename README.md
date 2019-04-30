# openwhyd
beginning a rec-sys for the music sharing platform Openwhyd

3 methods considered so far:

SIMPLE ALGO
an algorithm from first principles which simply matches users who have at least one song in common and then proposes to each
songs of the other

MATRIX FACTORISATION
uses ALS from the Implicit library and a system of matrices to suggest n number of songs for a given user. Appears to work

KNN
returns a predicted score of a user U when s/he is faced with a song S. Not clear how it's to be exploited.

I have tried and failed with some others:
Frequent Pattern Growth, Association Rule Mining, Bayesian Personalised Ranking
