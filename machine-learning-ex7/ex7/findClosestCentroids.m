function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:size(X,1)
    minDistortion = intmax();
    for j = 1:K
        % calculate the diff (1xn matrix - 1xn matrix) which results in another 1xn matrix
        diff = X(i,:) - centroids(j,:);
        % the magnitude of the 1xn matrix/vector is the sqrt of the sum of each squared component sqrt(a^2 + b^2).
        % since we square the magnitude in the projection error formula, it is just a^2 + b^2.
        % we can calculate this by multiplying the matrix with the transpose of itself
        testDistortion = diff*diff';
        if testDistortion < minDistortion
            minDistortion = testDistortion;
            idx(i) = j;
        end
    end
end

% =============================================================

end

