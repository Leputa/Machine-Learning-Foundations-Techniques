For Questions 16-20, you will play with the decision stump algorithm.

In class, we taught about the learning model of "positive and negative rays'' (which is simply one-dimensional perceptron) for one-dimensional data. The model contains hypotheses of the form:

hs,θ(x)=s*sign(x−θ).

The model is frequently named the "decision stump'' model and is one of the simplest learning models. As shown in class, for one-dimensional data, the VC dimension of the decision stump model is 2.

In fact, the decision stump model is one of the few models that we could easily minimize Ein efficiently by enumerating all possible thresholds. In particular, for N examples, there are at most 2N dichotomies (see page 22 of lecture 5 slides), and thus at most 2N different Ein values. We can then easily choose the dichotomy that leads to the lowest Ein, where ties an be broken by randomly choosing among the lowest Ein ones. The chosen dichotomy stands for a combination of some "spot" (range of θ) and s, and commonly the median of the range is chosen as the θ that realizes the dichotomy.