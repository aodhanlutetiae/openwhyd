# openwhyd
Building a rec-sys for the music sharing platform Openwhyd

MATRIX FACTORISATION
uses ALS from the Implicit library and a system of matrices to suggest n number of songs for a given user. Returns N number of recommended songs for a user, U.

KNN
returns a predicted score of a user U when s/he is faced with a song S. Not clear how it's to be exploited on a large scale.

SIMPLE ALGO
an algorithm from first principles which simply matches users who have at least one song in common and then proposes to each
songs of the other

OTHER FILES

CLEANING THE OPENWHYD DATALOG
About 3 of the 25 million records involve youtube links that are dead. This notebook shows how the original datalog was cleaned.

TIMESTAMP DATA
Notebook that manipulates the timestamp data that's included in the log and shows user engagement over the time period of the log.


I have tried and failed with some other algorithsm:
Frequent Pattern Growth, Association Rule Mining, Bayesian Personalised Ranking
