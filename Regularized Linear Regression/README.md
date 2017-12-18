13
Consider regularized linear regression (also called ridge regression) for classification

\mathbf{w}_{\text{reg}} = \mbox{argmin}_{\mathbf{w}} \left(\frac{\lambda}{N} \|\mathbf{w}\|^2 + \frac{1}{N} \|\mathbf{X}\mathbf{w}-\mathbf{y}\|^2\right) .

Run the algorithm on the following data set as D:

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_train.dat

and the following set for evaluating Eout:

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_test.dat

Because the data sets are for classification, please consider only the 0/1 error for all Questions below.

Let λ=10, which of the followings is the corresponding Ein and Eout?

