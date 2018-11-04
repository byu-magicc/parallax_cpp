# Parallax CPP Tests

Ways to show results
* Plot: Truth error over time & chierality
* Table: Overall average truth error & chierality
* Plot: Truth error vs. number of iterations (use for overall number of iterations)
* Plot: LMEDS error vs. number of iterations (use to show that a prior is very helpful)




Things to show (in order of appearance in paper)
Optimizer cost:
- This doesn't seem that important. Most people would wonder why we aren't just optimizing the cost function we picked.
- It's still interesting, but low priority

Consensus cost:
- This one is pretty important
- Must use error truth

GN vs. LM:
- Also important
- Must use error truth

LMEDS vs. RANSAC:
- Cool threshold plot!

Seed (importance of using recursive seeding)
- I think the graph we had was sufficient

Timing


Refinement






Note: Plots will slowly become outdated as we fine-tune parameters. For example, it still seems
unclear the exact number of minimum subsets we need. We are going to run all the GN/LM algorithms
with 100 for consistency, except when comparing directly to OpenCV_poly.