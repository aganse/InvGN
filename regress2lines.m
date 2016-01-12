function [mstar, Rstar, idiv, Gstar] = regress2lines(x,y)
% REGRESS2LINES - fit two cojoined straight lines to data points.
% This is a nonlinear estimation problem.  The data will be divided into two
% sets, each fitted with a line; the lines intersect between the data division.
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2010.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% IMPORTANT - You should be able to justify such a restricted choice of
% model as simply two straight lines.  Perhaps it makes more sense for you
% to instead fit a single, higher-degree polynomial (an easy linear problem)
% or exponential curve to your data.  To use this two-phase straight-line
% regression on your data you should be able to cite some theory for why
% you specifically want two intersecting straight lines as opposed to a
% more general model.
%
% Usage:   [m, R, idiv, G] = regress2lines(x,y);
%
%   Input - data column vectors x & y of equal length.
%        
%   Outputs - 
%      m = column vector of the 5 solution paramters [a1,b1,a2,b2,x0] ala
%             lines y=a1*x+b1 and y=a2*x+b2, intersecting at x0.
%      R = sum of sqrs of residuals of the solution
%      idiv = the index of the data point after which the data was divided.
%          x0 is between x(idiv) and x(idiv+1).
%      G = the linear problem matrix associated with the solution x0.
%             This might be used to calculate *some* stats of the solution.
%
% Since the problem as a whole is nonlinear, the full picture of the
% solution statistics is seen by Monte Carlo simulation, as demonstrated and
% explored in my MultiRegressLines.matlab package, available at
% https://www.github.com/aganse/MultiRegressLines.matlab


% More specific technical description:
%
% The problem is to solve:
%   Given xdata(1:n),ydata(1:n)
%   Find m={a1,b1,a2,b2,x0} such that we
%   Minimize data misfit in least-squares sense
%   ie min sum( ydata(1:istar)-a1*xdata(1:istar)-b1 )^2 +
%          sum( ydata(istar+1:end)-a2*xdata(istar+1:end)-b2 )^2
%   With constraint:  x(istar) <= x0=(b2-b1)/(a1-a2) <= x(istar+1)
% (ie x0 is the intersection of the two fit lines and is at the break in
% data)
%
% The problem as a whole is a nonlinear regression problem because of
% the parameter x0.  But it's really convenient that for a given constant
% value of x0 the problem becomes linear, on top of which each x0
% corresponds to a single division in the data points into two sets.
% (For each set we compute a standard linear fit, and x0 is the unique
% location at which the two lines intersect.)
% Given a value for x0, which tells the data points between which to split
% the data, we set up the linear problem as y=G*m, where m=[a1,b1,a2,b2].
% x0 is then computed from the elements of m as x0=(b2-b1)/(a1-a2).
%
% We loop over the data points: each time dividing them into two groups,
% recording the sum-of-squares-of-residuals R and the x0 value.  And after
% going over all the data points the solution is ideally the x0 and its
% associated a1,b1,a2,b2 which correspond to the minimum R value.  We do
% this "brute force" approach to the nonlinear problem rather than such
% often-seen methods like locally approximating the nonlinear function as
% linear and using derivatives, because this problem has discontinuities
% that mess up the differentiability at the data points.  And the search
% over all x0 is very small because it is discretized - there's only one
% x0 that corresponds to each data division, and so we only need to compute
% N-2 values of R to find that minimum, where N is the number of data points.
% Note there is a remote possibility that there will be two equal minima,
% ie that there are two equally worthy places to intersect two fit lines,
% so one should look for that when analyzing our R values and report both
% cases.  Please see a full lowdown at:
% http://staff.washington.edu/aganse/mpregression/index.html
% Especially note on that webpage the important caveats about when this
% script will and will not work.


N=length(y);
igood=zeros(N-2,1); x0vector=igood; Rvector=igood;  % creating col vectors
for i=2:N-1
    % set up the linear problem as y=G*m, where m=[a1,b1,a2,b2]:
    G(1:i,:)=[x(1:i),ones(i,1),zeros(i,2)];
    G(i+1:N,:)=[zeros(N-i,2),x(i+1:N),ones(N-i,1)];
    m=pinv(G'*G)*G'*y;  % least-squares solution to linear problem
    x0=(m(4)-m(2))/(m(1)-m(3));  % where the two fit lines intersect
    R=sum( (y-G*m).^2 );
%fprintf(1,'x(1)=%f, x0=%f, x(N)=%f\n',x(1),x0,x(N));
    if x0>=x(1) && x0<=x(N)
        igood(i-1)=1;
        x0vector(i-1)=x0;
        Rvector(i-1)=R;
    end
end

% Pick out the minimum R and its associated solution info:
%disp(num2str([(2:N-1)',igood,x0vector,Rvector]));  % (for debugging)
[Rstar,imin]=min(Rvector(find(igood==1)));
idiv=imin+1;
Gstar(1:idiv,:)=[x(1:idiv),ones(idiv,1),zeros(idiv,2)];
Gstar(idiv+1:N,:)=[zeros(N-idiv,2),x(idiv+1:N),ones(N-idiv,1)];
m=pinv(Gstar'*Gstar)*Gstar'*y;  % least-squares solution to linear problem
mstar=[m;(m(4)-m(2))/(m(1)-m(3))];




% Test data, etc...:
% N=40;
% x=sort(N*rand(N,1));
%
% noisy two lines:
% G(1:i,:)=[x(1:i),ones(i,1),zeros(i,2)];
% G(i+1:N,:)=[zeros(N-i,2),x(i+1:N),ones(N-i,1)];
% y=G*[4,3,-1,73]';
% yd=y+4*randn(N,1);
%
% noisy parabola:
% G=[x.^2/2, x, ones(N,1)];
% y=G*[4,4,4]';
% yd=y+10*randn(N,1);
% 
% plots:
% plot(x,y,'-',x,yd,'*')
% plot(x,y,'-',x,yd,'*',[x(1);m(5);m(5);x(end)],...
%     [m(1)*x(1)+m(2), m(1)*m(5)+m(2), m(3)*m(5)+m(4), m(3)*x(end)+m(4)],'-+')
