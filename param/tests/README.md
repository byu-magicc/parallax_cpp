# Parallax CPP Tests

Ways to show results
* Plot: Truth error over time & chierality
* Table: Overall average truth error & chierality
* Plot: Truth error vs. number of iterations (use for overall number of iterations)
* Plot: LMEDS error vs. number of iterations (use to show that a prior is very helpful)






This folder contains matlab scripts to plot the results for the iterative 5-point algorithm journal paper.

* `lambda.m` - What should the Levengerg-Marquardt lambda start off with?
* `scoring_cost.m` - Is algebraic, single, or sampson scoring cost better?
* `ransac.m` - RANSAC threshold and LMEDS vs. RANSAC.
* `chierality.m` - Best method for pose disambiguation.

* `consensus_alg.m` - Which consensus algorithm to use?



