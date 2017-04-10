function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% At the time that featureNormalize.m is called, 
% the extra column of 1?s corresponding to x0 = 1 
% has not yet been added to X 

% You need to set these values correctly
X_norm = zeros(size(X));
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
j = size(X,2);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%

mu = mean(X); sigma=std(X);

for i = 1:j
    X_norm(:,i) = (X(:,i)-mu(i))/sigma(i);
end







% ============================================================

end
