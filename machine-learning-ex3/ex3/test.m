
%theta = [-2; -1; 1; 2];
%X = [ones(5,1) reshape(1:15,5,3)/10]
%y = [1;0;1;0;1] >= 0.5       % creates a logical array
%lambda = 3;
%[J grad] = lrCostFunction(theta, X, y, lambda)
%J;
%grad;
%disp(size(grad));

%all_theta = [1 -6 3; -2 4 -3];
%X = [1 7; 4 5; 7 8; 1 4];
%predictOneVsAll(all_theta, X)

Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X = reshape(sin(1:16), 8, 2);
p = predict(Theta1, Theta2, X)


%  X = [ones(20,1) (exp(1) * sin(1:1:20))' (exp(0.5) * cos(1:1:20))'];
%  y = sin(X(:,1) + X(:,2)) > 0;
%  Xm = [ -1 -1 ; -1 -2 ; -2 -1 ; -2 -2 ; ...
%          1 1 ;  1 2 ;  2 1 ; 2 2 ; ...
%         -1 1 ;  -1 2 ;  -2 1 ; -2 2 ; ...
%          1 -1 ; 1 -2 ;  -2 -1 ; -2 -2 ];
%  ym = [ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ]';
%  t1 = sin(reshape(1:2:24, 4, 3));
%  t2 = cos(reshape(1:2:40, 4, 5));
%
%  [J, grad] = costFunctionReg([0.25 0.5 -0.5]', X, y, 0.1);
%  printf('J: %0.5f \n', J);
%  printf('grad: %0.5f \n', grad);
%
%  [J, grad] = lrCostFunction([0.25 0.5 -0.5]', X, y, 0.1);
%  printf('J: %0.5f \n', J);
%  printf('grad: %0.5f \n', grad);

