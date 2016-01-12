function testFindiffStepsize(fwdprob,mtest,mplot,varargin)
% TESTFINDIFFSTEPSIZE - Analyze step size choice for 1st order finite differences
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2010.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Usage:
% testFindiffStepsize(fwdprob,mtest,mplot,varargin);


ind=0:1:12;  % indices defining the step sizes  (0:1:20)

for i=ind
  epsf=10^(-i);
  dx=sqrt(epsf); % true for fwd diffs when vars scaled to O(1)
  if i<0, ii=['m',num2str(-i)]; else ii=num2str(i); end;

  disp(['  computing Dfwd(h) for epsr=10^(-' ii ')...']);
  eval(['J' ii '=jacfwd(fwdprob,mtest,-dx,0,1,varargin{:});']);
  eval(['J' ii '_2=jacfwd(fwdprob,mtest,-2*dx,0,1,varargin{:});']);
  eval(['D' ii '=J' ii '_2-J' ii ';']);
end
clear J*;

% plotting:
for m=mplot
  figure;
  h=sqrt(10.^(-ind));
  d1=1; d2=6; dincr=1;  % use all 6 data pts
  c=ones(1,ceil((d2-d1+1)/dincr));  % just making as many ones as data pts
  
  % first form the vectors to plot, in the following format:
  % x=[h(1)*c;h(2)*c;...]
  % y=[D0(d1:d2,m)';D1(d1:d2,m)';...]
  x=[]; y=[];
  for j=1:length(ind)
    x=[x;h(j)*c];
    eval([ 'y=[y;abs(D' num2str(ind(j)) '(d1:dincr:d2,m))''];']);
    %eval([ 'y=[y;D' num2str(ind(j)) '(d1:dincr:d2,m)''];']);
  end
  
  loglog(x,y,'*-');
  axis tight
  grid on
  t=title(['F=' fwdprob ', param#' num2str(m)]);
  set(t,'Interpreter','none');  % prevent "_" characters in title causing subscript
  xlabel('finite difference stepsize h');
  ylabel('\partial{F}_{(2h)} - \partial{F}_{(h)}');
end
