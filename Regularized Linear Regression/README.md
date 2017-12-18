13

Consider regularized linear regression (also called ridge regression) for classification

wreg=argminw(λN∥w∥2+1N∥Xw−y∥2).

Run the algorithm on the following data set as D:

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_train.dat

and the following set for evaluating Eout:

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_test.dat

Because the data sets are for classification, please consider only the 0/1 error for all Questions below.

Let λ=10, which of the followings is the corresponding Ein and Eout?

14

Following the previous Question, aong log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Ein? Compute λ and its corresponding Ein and Eout then select the closest answer. Break the tie by selecting the largest λ.

15

Following the previous Question, among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Eout? Compute λ and the corresponding Ein and Eout then select the closest answer. Break the tie by selecting the largest λ.

16

Now split the given training examples in D to the first 120 examples for Dtrain and 80 for Dval. \textit{Ideally, you should randomly do the 120/80 split. Because the given examples are already randomly permuted, however, we would use a fixed split for the purpose of this problem.}

Run the algorithm on Dtrain to get g−λ, and validate g−λ with Dval. Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Etrain(g−λ)? Compute λ and the corresponding Etrain(g−λ), Eval(g−λ) and Eout(g−λ) then select the closet answer. Break the tie by selecting the largest λ.

17

Following the previous Question, among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Eval(g−λ)? Compute λ and the corresponding Etrain(g−λ), Eval(g−λ) and Eout(g−λ) then select the closet answer. Break the tie by selecting the largest λ.

18

Run the algorithm with the optimal λ of the previous Question on the whole D to get gλ. Compute Ein(gλ) and Eout(gλ) then select the closet answer.

19

For Questions 19-20, split the given training examples in D to five folds, the first 40 being fold 1, the next 40 being fold 2, and so on. Again, we take a fixed split because the given examples are already randomly permuted.

Among log10λ={2,1,0,−1,…,−8,−9,−10}. What is the λ with the minimum Ecv, where Ecv comes from the five folds defined above? Compute λ and the corresponding Ecv then select the closet answer. Break the tie by selecting the largest λ.

20

Run the algorithm with the optimal λ of the previous problem on the whole D to get gλ. Compute Ein(gλ) and Eout(gλ) then select the closet answer.