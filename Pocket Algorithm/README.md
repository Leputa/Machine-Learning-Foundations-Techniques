18.

Next, we play with the pocket algorithm. Modify your PLA in Question 16 to visit examples purely randomly, and then add the "pocket" steps to the algorithm. We will use

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_train.dat

as the training data set D, and

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_test.dat

as the test set for "verifying'' the g returned by your algorithm (see lecture 4 about verifying). The sets are of the same format as the previous one. Run the pocket algorithm with a total of 50 updates on D , and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

19.

Modify your algorithm in Question 18 to return w50 (the PLA vector after 50 updates) instead of w^ (the pocket vector) after 50 updates.

Run the modified algorithm on D, and verify the performance using the test set.

Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?

20.

Modify your algorithm in Question 18 to run for 100 updates instead of 50, and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?