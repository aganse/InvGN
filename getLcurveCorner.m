function corner=getLcurveCorner(rough,misfit,method)
% GETLCURVECORNER - Give index of Lcurve corner based on ROUGH and MISFIT values
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2010.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Usage:  corner=getLcurveCorner(rough,misfit)
% 
% ROUGH and MISFIT are the equal-length vectors with the Lcurve axes values.
% CORNER is the index of of the corner.
% METHOD (optional - its default is "derivative-based") can be one of:
%   0 = derivative-based
%   1 = two-phase linear regression to find intersection point of two lines
%
% Uses two-phase linear regression in regress2lines() to find the knee;
% suggested that ROUGH and MISFIT axes are squares of the respective two-norms.

if nargin<3, method=0; end

  % Scale Lcurve so axes same magnitude:
  % (and the (:) ensures they are column vectors...)
  x=rough(:)/max(rough); y=misfit(:)/max(misfit);
  % Rotate Lcurve by 45deg to stabilize:
  theta=pi/180*45;
  rr=[cos(theta) -sin(theta); cos(theta) sin(theta)];
  xxyy=rr*[x';y'];
  xx=xxyy(1,:)'; yy=xxyy(2,:)';


if method==0
  % use derivatives:
  [dummy,corner]=max(diff(diff(yy)./diff(xx)));
elseif method==1
  % use two-phase linear regression:
  addpath('~/APLUW/src/matlab/AGmultiphase_linreg/');
  [dummy,dummy,corner] = regress2lines(xx,yy);
elseif method==2
  % just find min of rotated lcurve:
  [dummy,corner]=min(yy);
end

%plot(rough.^2,misfit.^2,'*-');
%hold on;
%plot(rough(corner).^2,misfit(corner).^2,'r*');

% example derivative-based method - works great sometimes...
%e=log10(normrough(:,iters)); r=log10(normmisfit(:,iters));
%[dummy,corner]=max(diff(diff(r)./diff(e)));  % 2nd deriv to find lcurve knee
