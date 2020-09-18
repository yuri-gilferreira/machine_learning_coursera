function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_list = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_list = [0.01,0.03,0.1,0.3,1,3,10,30];
length_c = length(C_list);
length_sigma = length(sigma_list);
cv_validation_error = zeros(length_c,length_sigma);
for i = 1:length(C_list)
  for j = 1:length(sigma_list)
    C_test = C_list(i);
    sigma_test = sigma_list(j);
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));    
    predictions_cv = svmPredict(model, Xval);
    cv_validation_error(i,j) = mean(double(predictions_cv ~= yval));
  endfor
  endfor
cv_validation_error
[minval, row] = min(min(cv_validation_error,[],2));
[minval, col] = min(min(cv_validation_error,[],1));

C = C_list(row);
sigma = sigma_list(col);








% =========================================================================

end
