18

For Questions 18-20, you will play with logistic regression. Please use the following set for training:

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat

and the following set for testing:

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat

Implement the fixed learning rate gradient descent algorithm for logistic regression. Run the algorithm with η=0.001 and T=2000. What is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

19

Implement the fixed learning rate gradient descent algorithm for logistic regression. Run the algorithm with η=0.01 and T=2000, what is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

20

Implement the fixed learning rate stochastic gradient descent algorithm for logistic regression. Instead of randomly choosing n in each iteration, please simply pick the example with the cyclic order n=1,2,…,N,1,2,….

Run the algorithm with η=0.001 and T=2000. What is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?