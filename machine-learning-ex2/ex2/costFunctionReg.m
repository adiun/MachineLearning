function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
sumGrad = 0;
for i = 1:m
  z = theta' * X(i,:)';
  h = sigmoid(z);
  sum = sum + ((-y(i) * log(h)) - ((1 - y(i)) * (log(1 - h))));
  sumGrad = sumGrad + ((h - y(i)) * X(i,:)');
endfor

J = (1/m) * sum;

sumReg = 0;
numFeatures = size(X,2);
for j = 2:numFeatures
  sumReg = sumReg + (theta(j)'^2);
endfor

J = J + ((lambda / (2 * m)) * sumReg);
thetaSize = size(theta,1);
%disp(size(sumGrad(2:thetaSize)))
%disp(size(theta(2:thetaSize)))
grad1 = ((1/m) * sumGrad(1));
grad2 = ((1/m) * sumGrad(2:end)) + ((lambda / m) * theta(2:end));
grad = [grad1; grad2];
%grad = ((1/m) * sumGrad) + ((lambda / m) * theta);

% =============================================================

end
