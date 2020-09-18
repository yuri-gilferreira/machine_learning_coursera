function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J_unreg = 1/(2*m) * sum(((X*theta - y) ).^2);
reg_term = sum(lambda/(2*m) * theta(2:end).^2);
J = J_unreg + reg_term;


grad = (sum(((theta'*X')' - y).*X,1)*(1/m))';
reg_theta = (lambda/m) * theta;
reg_theta(1) = 0;
grad = grad + reg_theta;




%{
x_test = [ones(m, 1) X];

temp_sum = 0 ;
for i = 1:m
  temp_sum = temp_sum + (theta(1)*x_test(i,1) - y(i));
endfor
temp_sum = temp_sum/m;
%}


% =========================================================================

grad = grad(:);

end
