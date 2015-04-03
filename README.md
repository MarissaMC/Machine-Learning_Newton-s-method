Newton's method
=====================
The code written with Matlab.

Consider two cases: unregularized and regularized

* **Newton_method**
   Input: train_data, train_label,test_data,test_label,w0,b0
   Output: graph of Unregularized Cross-Entropy value, L2-norm value 

* **Newton_method_regularized**
   Input: train_data, train_label,test_data,test_label,w0,b0
   Output: graph of regularized Cross-Entropy value, L2-norm value 

Usage
-----------------
   Newton_method(train,i_train_label,test,test_label,w0,b0);
   Newton_method_regularized(train,i_train_label,test,test_label,w0,b0);
