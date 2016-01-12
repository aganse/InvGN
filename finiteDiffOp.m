function L=finiteDiffOp(order,segmentlengths,dx,BCoption)
% FINITEDIFFOP - Return an ORDERth order finite difference operator matrix
% with "kinks" in it to break it into segments of SEGMENTLENGTHS to accommodate
% discontinuities in a profile model, or multiple profiles concatenated together
% (e.g. to avoid regularizing across the profile boundaries).
% Various boundary conditions for the operator may be specified, or none at all,
% according to the codes in BCoption. (NOTE BCOPTION ONLY PARTIALLY IMPLEMENTED)
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, 2010.
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
%
% Usage:
% L=finiteDiffOp(order,segmentlengths,dx,BCoption);
%
% Inputs:
%          order = order of fin diff operator (=0,1,2)
% segmentlengths = vector containing number of model elements between each
%                  discontinuity, or can be simply a scalar value with number
%                  of total model elements if no discontinuities in model.
%                  Note this could equivalently be used when model vector is
%                  a concatenation of multiple profiles such that you don't
%                  want to smooth boundary from one profile to next.
%             dx = delta x between model parameters in continuous profile.
%   BCoption: NOTE: OPTS 2-5 REQUIRE CODE FIX AT BTM OF SCRIPT, BUT OPTS 0-1 OK.
%      0     = no additional bdy conds.
%      1     = prefer 0 gradient between last microlayer and basement,
%              trying to keep vels equal so no reflection from basement.
%      2     = Constrain f'' to equal zero at the endpoints, so that there is
%              zero curvature there.  L not full rank here.
%      3     = Extends minimum curvature integral from -inf to +inf, forcing 
%              f=0 at all nodes outside grid.  f'' at boundary and first node
%              outside grid can have non-zero curvature.  Note results using
%              this BC can be similar to #4 because of the nearby f=0.  But L
%              is full rank here.
%      4     = Constrain f' to equal zero at the endpoints, so that there is
%              zero gradient there.  L not full rank here.
%      5     = Constrain f itself to equal zero at the endpoints.  L full rank.
%
% Outputs:
%   L = the finite diff operator matrix 


if order==0
    % "Zeroth order Tikhonov regularization", also called "ridge regression".
    % Note no segment bdys are handled here because 0th order regu doesn't care.
    L=eye(sum(segmentlengths));
    
elseif order==1
    % "first order Tikhonov regularization" for total variation regularization

    L=[];
    for i=1:length(segmentlengths)
      M=segmentlengths(i);
      L1=-1*diag(ones(M,1)) + diag(ones(M-1,1),+1);  % fwd diffs : O(dx)
      %L1=diag(ones(M-1,1),1) - diag(ones(M-1,1),-1); % cen diffs : O(dx^2)
      L1=L1(1:(end-1),:);  % diag() by itself didn't give quite what we wanted -
                           % it gave a square matrix with implicit bdy conds -
                           % so we're just removing top and bottom rows with BCs.
                           % Note nonsquare now.
      L=blkdiag(L,L1);
    end

    % NOTE dx presently not implemented but if reimplementing note dx is now a vector.

    % Following is dx normalization such that L*m = 1st deriv of m.
    % Only really makes sense to do this when not including other constraints
    % or segments in problem, otherwise easier to just leave off:
    %L=L/dx;    % for forward diffs to O(dx)
    %L=L/dx/2;  % for central diffs to O(dx^2)

elseif order==2
    % "2nd order Tikhonov regularization" for Occam's (smoothing) regularization

    % central diffs : O(dx^2) :
    L=[];
    for i=1:length(segmentlengths)
      M=segmentlengths(i);
      L1=-2*diag(ones(M,1)) + diag(ones(M-1,1),1) + diag(ones(M-1,1),-1);
      if BCoption==1
        L1(1,1)=-1; L1(end,end)=-1;
      elseif BCoption==0
        L1=L1(2:end-1,:);  % diag() by itself didn't give quite what we wanted -
                           % it gave a square matrix with implicit bdy conds -
                           % so we're just removing top and bottom rows with BCs.
                           % Note nonsquare now.
      end
      L=blkdiag(L,L1);
    end

    % NOTE dx presently not implemented but if reimplementing note dx is now a vector.

    % Following is dx normalization such that L*m = 2nd deriv of m.
    % Only really makes sense to do this when not including other constraints
    %or segments in problem, otherwise easier to just leave off:
    %L1=L1/dx^2;
    %L1=L1/sqrt(M-2);  % or here we normalize by #pts-2 (ie subtracting endpts)
                       % to address the finite evaluation domain.
                       % Now the units of L1*vector are M/X^2, where M are
                       % the units of model and X are the units of the axis
                       % the model is discretized over.

end



if 0
% Apply boundary conditions based on input options:
% Note if BCoption==0 then all this is skipped...
if BCoption==1, L(end+1,end-1:end)=[-1,1]; end;  % for my microlayers on top
                                                 % of lower halfspace case -
                                                 % prefer zero-gradient between
                                                 % last microlayer and halfspace
                                                 % to prevent reflections there
if BCoption==2, topleftcorner=[0 0 0 0]; btmrightcorner=[0 0 0 0]; end;
if BCoption==3, topleftcorner=[-2 1 0 0]; btmrightcorner=[0 0 1 -2]; end;
if BCoption==4, topleftcorner=[-2 2 0 0]; btmrightcorner=[0 0 2 -2]; end;
if BCoption==5, topleftcorner=[-2 0 0 0]; btmrightcorner=[0 0 0 -2]; end;

if BCoption>1,
  % WARNING: FIXME: this part is not ready and must be correctly implemented:
  % need to first prepend a new row to top of L and append a new row to btm of L, then:
  %L(1,1:4)=topleftcorner;
  %L(M,M-3:M)=btmrightcorner;
end

end
