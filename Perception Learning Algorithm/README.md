For Questions 15-20, you will play with PLA and pocket algorithm. First, we use an artificial data set to study PLA. The data set is in

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat

Each line of the data set contains one (xn,yn) with xn∈R4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.

Please initialize your algorithm with w=0 and take sign(0) as −1. Please always remember to add x0=1 to each xn.


15.
Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?


16.
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm. Run the algorithm on the data set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?


17.
Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm, while changing the update rule to be

wt+1←wt+ηyn(t)xn(t)

with η=0.5. Note that your PLA in the previous Question corresponds to η=1. Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts?