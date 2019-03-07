clear all, close all, clc

% Calculates the mean average precision for the two datasets provided in
% task 1d

xi = [0:0.1:1]';

prec1 = [1.0, 1.0, 1.0, 0.5, 0.20]';
rec1 =  [0.05, 0.1, 0.4, 0.7, 1.0]';

prec1_i = interp1q(rec1, prec1, xi);
prec1_i(1) = 1; %Fix first value, not covered by interp1q()

prec2 = [1.0, 0.80, 0.60, 0.5, 0.20]';
rec2 = [0.3, 0.4, 0.5, 0.7, 1.0]';

prec2_i = interp1q(rec2, prec2, xi);
prec2_i(1) = 1; %Fix first value, not covered by interp1q()

%Finding max(p(r)) function
for i = 1:length(xi)
   prec1_i(i) = max(prec1_i(i:length(prec1_i)));
   prec2_i(i) = max(prec2_i(i:length(prec2_i)));
end

AP1 = mean(prec1_i);
AP2 = mean(prec2_i);