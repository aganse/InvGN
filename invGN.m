function [m,normdm,normmisfit,normrough,lambda,dpred,G,C,R,Ddiag,i_conv]=...
   invGN(fwdfunc,derivfunc,epsr,dmeas,covd,lb,ub,m0,Lin,lambdaIn,useMpref,...
         maxiters,verbose,fid,varargin)

% INVGN - Compute Gauss-Newton nonlinear inversion, version 1.0
% Calculate Tikhonov-regularized, Gauss-Newton nonlinear iterated inversion
% to solve the damped nonlinear least squares problem.
%
% by Andrew Ganse, Applied Physics Laboratory, University of Washington, Seattle.  
% Contact at andy@ganse.org , http://research.ganse.org
% Copyright (c) 2015, University of Washington, via 3-clause BSD license.  
% See LICENSE.txt for full license statement.  
% See manpage.txt for full description and usage instructions.
%
% Usage summary:
% [m,normdm,normmisfit,normrough,lambda,dpred,G,C,R,Ddiag,i_conv] = ...
% invGN(fwdfunc,derivfunc,epsr,dmeas,covd,lb,ub,m0,L,lambda,useMpref,maxiters,...
%       verbose,fid,vargin);


% Some more programattic run options (so left off argument list):
usingOctave=0;       % 1=octave, 0=matlab (affects form of just a few commands).
use_central_diffs=0; % 0=no use fwd finite diffs, 1=yes use centr finite diffs.
                     % central diffs are more accurate but twice the calc time.
                     % In practice this often comes down to a tradeoff between
                     % calc time for one jacobian matrix vs number of iterations
                     % (more accurate derivs may make for fewer iterations).
use_dist_jaccalc=0;  % 0=compute all columns of Jacobian finite diffs one at 
                     % a time in sequence.  (only recommended for few params.)
                     % 1=compute all columns of Jacobian finite diffs in nproc
                     % simulaaneous instances of octave/matlab on same host.
                     % (nproc is set in jacfwdN.m and jaccenN.m, typically 4or8)
                     % 2=distribute computation of Jacobian finite diff columns
                     % over all nodes of bluewater cluster.
saveIntermediate=0;  % save intermediate results into a file mid-script in case
                     % of crash, for slow-running fwd problems and/or debugging.
                     % Filename is invGN_<pid>.mat.
                     % 0=none, 1=at end of each lambda loop, 2=at end of each
                     % iteration too.
keepalliters=0;      % 1=add a dimension to the output matrices to save all
                     % iters while debugging/studying a new inverse problem (but
                     % watch out - this can potentially eat up a LOT of memory).
                     % 0=only keep the info for the last iteration, ie solution
                     % point, but still keep normdm iterations (default).
reuse_first_G=1;     % save compute time by not recomuting the first (same)
                     % jacobian matrix for each lambda (they're same because
                     % same minit for all lambdas)
check_objfn=0;       % Ensure the objective function is decreasing with each
                     % iteration; if not then halve the step and check again
                     % till satisfied.  Presently this is set up to only kick in
                     % after "convergence" (based on dm); see next option...
check_objfn_iters=0 ;% If using check_objfn=1, then after hitting "convergence"
                     % (based on dm), continue this many iterations with the
                     % decreasing objective function constraint.  (This option
                     % is ignored if check_objfn=0.)
check_bnds=1;        % Ensure the iteration does not step out of the bounds;
                     % if so then halve the step and check again till satisfied.
outbnds_method=0;    % If using check_bnds=1, use one of the following methods:
                     % if 0 revert to half^n step in same direction for nth step.
                     % if 1 use reflection approach to handle overstepping bnds,
                     % if 2 set param to bounds if oversteps bnds.
maxstephalves=10;    % max # times to halve step when using either:
                     % (check_bnds AND outbnds_method=0) OR check_objfn.
warningsoff=1;       % Regardless of verbosity level, turn off calculational
                     % warnings; currently this is only the high cond# warning
                     % and is done so that invGN can give more specific warning.


warning('off','MATLAB:nearlySingularMatrix');  % instead caught manually and put in
                                               % better format for this script's output


% Dimensions and preliminaries:
ndata=length(dmeas);
nparams=size(m0,1);
if lambdaIn(1)<0
   nlambdas=-lambdaIn(1);
   if length(lambdaIn)==3
     maxlambdaexp=lambdaIn(2);
     minlambdaexp=lambdaIn(3);
   else
     % otherwise use these default values:
     maxlambdaexp=3;
     minlambdaexp=-3;
   end
   if verbose>1
     fprintf(fid,'Max/min exponent values for Lcurve: %d, %d...\n',maxlambdaexp,minlambdaexp);
     fprintf(fid,'Lambda values for Lcurve will be determined during 1st iteration...\n');
     if usingOctave, fflush(fid); end
   end
else
   nlambdas=length(lambdaIn);
   lambda=lambdaIn;
   if verbose>1
     fprintf(fid,'Using lambda values for Lcurve specified by user...\n');
     if usingOctave, fflush(fid); end
   end
   % FIXME: add a check that all lambdas are positive...
end;
if verbose>1
  fprintf(fid,'Number of lambdas (Lcurve points): %d\n',nlambdas);
  fprintf(fid,'Max number of GN iterations at each lambda: %d\n',maxiters);
  fprintf(fid,'Number of model parameters: %d\n',nparams);
  fprintf(fid,'Number of data points: %d\n',ndata);
  if outbnds_method==0
     fprintf(fid,'Out-of-bounds steps will revert to half-step in same direction...\n');
  elseif outbnds_method==1
     fprintf(fid,'Out-of-bounds steps will reflect from bounds...\n');
  elseif outbnds_method==2
     fprintf(fid,'Out-of-bounds steps will stop at bounds...\n');
  end
  if usingOctave, fflush(fid); end
end
if size(m0,2)>1 && nlambdas~=size(m0,2)
   fprintf(fid,'invGN: error: numcols of m0 not equal to numlambdas.\n');
   if usingOctave, fflush(fid); end
   return;
end;
if usingOctave
  pid=getpid();  % this is a built-in function in Octave; for Matlab will need external getpid.m script
else
  pid=12345;
end


% Preallocate main (large) storage matrices:
if keepalliters
   m=zeros(nparams,nlambdas,maxiters+1);  % maxiters+1 because first iter holds init model
   dm=zeros(nparams,nlambdas,maxiters);
   normdm=zeros(nlambdas,maxiters);
   normmisfit=zeros(nlambdas,maxiters);
   normrough=zeros(nlambdas,maxiters);
   dpred=zeros(ndata,nlambdas,maxiters);
   residshat=zeros(ndata,nlambdas,maxiters);
else
   m=zeros(nparams,nlambdas);
   dm=zeros(nparams,nlambdas);
   %normdm=zeros(nlambdas,1);
   normdm=zeros(nlambdas,maxiters);
   normmisfit=zeros(nlambdas,1);
   normrough=zeros(nlambdas,1);
   dpred=zeros(ndata,nlambdas);
   residshat=zeros(ndata,nlambdas);
end
% An extra dimension of nlambdas on each of the following matrix variables will
% allow us to keep the matrix from the last iteration (hopefully equaling the
% soln point) for each lambda, so we can study solution sensitivities.  If
% that's taking too much mem then we can remove that dim.  Note before soln
% point for given lambda is reached, these matrices simply hold the value for
% the current iteration.
G=zeros(ndata,nparams,nlambdas);
C=zeros(nparams,nparams,nlambdas);
R=zeros(nparams,nparams,nlambdas);
Ddiag=zeros(ndata,nlambdas);
i_conv=zeros(nlambdas,1);

% Compute invsqrtcovd once here since it's expensive to compute (for matrices):
if size(covd,2)==1  % covd is either scalar or vector

  if size(covd,1)==1  % it's a scalar variance
    if verbose>1
      fprintf(fid,'Measurement noise variance specified as scalar Cd...\n');
      if usingOctave, fflush(fid); end
    end
    if covd<0
      invsqrtcovd=-covd*ones(ndata,1);
    else
      invsqrtcovd=covd^(-1/2)*ones(ndata,1);
    end

  else  % it's a column vector = diag of cov matrix
    if verbose>1
      fprintf(fid,'Measurement noise variances specified = diag(covmatrix)...\n');
      if usingOctave, fflush(fid); end
    end
    if covd(1,1)<0  % minus sign (in arglist as "-covd") flags that it's really
                   % the invcond. (this works since diagonal of cov matrix must
                   % be non-neg.)  Useful e.g. when implementing a data mask via
                   % zeroes in the covd diagonal.
      invsqrtcovd=-covd;  % already inverted and sqrt'd, just removing "-" flag
    else
      invsqrtcovd=covd.^(-1/2);  % otherwise it really is the variances; convert to invstdvs
    end
  end

else  % then covd is full covariance matrix

   if verbose>1
     fprintf(fid,'Full measurement noise cov matrix specified; computing invsqrtcovd...\n');
     if usingOctave, fflush(fid); end
   end;
   if covd(1,1)<1  % minus sign (in arglist as "-covd") flags that it's really the invcond.
                   % (this works since diagonal of cov matrix must be non-neg.)
                   % useful e.g. when implementing a data mask via zeroes in the covd diagonal.
     invsqrtcovd=-covd;  % it's already inverted and sqrt'd, just removing the "-" flag...
   else
     invsqrtcovd=covd^(-1/2);
   end
end

% By default, do 2nd-order Tikh regularization, ie smoothing based regu.
% (IDEA: if I allowed for Lin to be a vector - must be done for finiteDiffOp
% too - I could specify where discontinuities are that way... better yet would
% be an option to include discontinuity estimation with the rest, but I think
% that'll significantly change the form of the dm solution expressions.)

% Auto-create "roughening" matrix L (2nd order finite diff operator matrix)
% with no discontinuities or bdy conds:
if length(Lin)==1 && size(m0,1)>=8  % the >=8 is because m is supposed to be
                                      % a discretized approximation to a
                                      % continuous function.  Hard to imagine
                                      % that with length(m)<8 !
   % in this case, Lin is scalar and meant to be number of profile segments:
   dx=sqrt(epsr);  % (assumes that model vector elements are of order unity)
   L=finiteDiffOp(2,nparams*ones(Lin,1),dx,0); %*sqrt(dx);  % non-sqr L, ie no bdy conds.
   % finiteDiffOp() returns a findiff operator, eg L=1/deltax^2*[1 -2 1...].
   % The *sqrt(dx) in the L= line above accommodates the application of the
   % finitediffs in the continuous integral norm on m:
   %   integ{(m'')^2}dx -> integ{(Lm)^2}dx -> (Lm)'*(Lm)*deltax
   %     = m'*(L'*L*deltax)*m = m'*Lhat'*Lhat*m, where Lhat=L*sqrt(deltax).
   % Rather than create a new var Lhat, I just included sqrt(dx) above.
   %
   % FIXME: temporarily commented out the sqrt(dx) part for num stability concerns...
elseif length(Lin)~=1 && size(m0,1)<8 && norm(Lin)~=0 && useMpref<2
   % catch the error case
   fprintf(fid,'invGN: error: if using default L matrix, must have length(m)>=8.\n');
   if usingOctave, fflush(fid); end
   return;  % errors out & dies.
else  
   % ie Lin is actually a user-supplied finite diff operator matrix L
   if norm(Lin)==0 && verbose>1
     fprintf(fid,'Noting L matrix of zeros to signal unregularized parameter estimation.\n');
     if usingOctave, fflush(fid); end
   end;
   L=Lin;
end


if size(m0,2)==1  % ie a single m0 col vector independent of lambda
   % Put m0 into first slot of m results matrix for each lambda (repetitive but more intuitive)
   if verbose>1
     fprintf(fid,'Populating initial model guesses for each lambda from m0 since all same...\n');
     if usingOctave, fflush(fid); end
   end;
   m(:,:,1)=m0*ones(1,nlambdas);  % ie make nlambdas cols of repeated m0 col vectors, equiv to repmat(m0,1,nlambdas)
else
   % In this case the initial model guess is diff't for each lambda, so m0 is a matrix.
   if verbose>1
     fprintf(fid,'Populating initial model guesses for each lambda from columns of m0...\n');
     if usingOctave, fflush(fid); end
   end;
   m(:,:,1)=m0;
end;


for l=1:nlambdas     % regularization tradeoff parameters lambda
   if verbose && l==1 && lambdaIn(1)<0  % lambdaIn(1)<0 tells us we must auto-find lambdas
      % won't know lambda values yet since based on G in 1st iteration:
      fprintf(fid,'Lambda(%d)  [value determined below in 1st iteration]...\n',l);
      if usingOctave, fflush(fid); end
   elseif verbose
      % here we know lambda values either from 1st iteration or from input arg:
      fprintf(fid,'Lambda(%d)=%g...\n',l,lambda(l));
      if usingOctave, fflush(fid); end
   end;
converged=0;
for iter=1:maxiters  % gauss-newton iterations for a given lambda
if verbose
  fprintf(fid,'  GN iteration #%d...\n',iter);
  if usingOctave, fflush(fid); end
end;


if verbose>1, tic; end;

if keepalliters, i=iter; else i=1; end;  % for saving all vs. just one iteration

% Compute local jacobian matrix of derivs G and local predicted data:
if isempty(derivfunc)
    % Compute finite diffs:
    if l==1 || iter>1 || size(m0,2)>1 || ~reuse_first_G
       if length(m0)<=2 || use_dist_jaccalc==0
          if use_central_diffs
             if verbose>1
               fprintf(fid,'    Computing finite diffs using jaccen() (single proc on local host)...\n');
               if usingOctave, fflush(fid); end
             end;
             [G(:,:,l),dpred(:,l,i)] = ...
                jaccen(fwdfunc,m(:,l,i),epsr,verbose,fid,varargin{:});
          else
             if verbose>1
               fprintf(fid,'    Computing finite diffs using jacfwd() (single proc on local host)...\n');
               if usingOctave, fflush(fid); end
             end;
             [G(:,:,l),dpred(:,l,i)] = ...
                jacfwd(fwdfunc,m(:,l,i),epsr,verbose,fid,varargin{:});
          end
       elseif use_dist_jaccalc==1
          if use_central_diffs
             fprintf(fid,'ERROR:  invGN:  locally-computed central-diffs is specified for calculating Jacobian\n');
             fprintf(fid,'  but this is not yet implemented in the code.  You could choose distributed-calc\n');
             fprintf(fid,'  central diffs, or locally-computed forward diffs at top of invGN...\n');
             return;
          else
             if verbose>1
                fprintf(fid,'    Computing finite diffs using jacfwdN() (distributed among procs of local host)...\n');
                if usingOctave, fflush(fid); end
             end;
             [G(:,:,l),dpred(:,l,i)] = ...
                jacfwdN(fwdfunc,m(:,l,i),epsr,verbose,fid,varargin{:});
          end
       elseif use_dist_jaccalc==2
          if use_central_diffs
            if verbose>1, fprintf(fid,'    Computing finite diffs using jaccendist() (network distributed calculation)...\n'); end;
            [G(:,:,l),dpred(:,l,i)] = ...
               jaccendist(fwdfunc,m(:,l,i),epsr,verbose,fid,varargin{:});
          else
            if verbose>1, fprintf(fid,'    Computing finite diffs using jacfwddist() (network distributed calculation)...\n'); end;
            [G(:,:,l),dpred(:,l,i)] = ...
               jacfwddist(fwdfunc,m(:,l,i),epsr,verbose,fid,varargin{:});
          end
       end
       if saveIntermediate == 2 
          % save expensive derivs immediately in case of crash before end of loop
          eval(['save -v7 invGN_l' num2str(l) '_i' num2str(i) '.mat']);
          if verbose>1
             fprintf(fid,'    saving intermediate workspace to file invGN_%d.mat...\n',pid);
             if usingOctave, fflush(fid); end
          end;
          eval(['save -v7 /tmp/invGN_' num2str(pid) '.mat']);
       end
       if nlambdas>1 && l==1 && iter==1
          % since the first m vector for every lambda is the same (=m0),
          % save the first G matrix so we don't unnecessarily recompute it for
          % every new lambda:
          if verbose>1
            fprintf(fid,'    Saving finite diffs from l=1,i=1 to avoid recomputing for l>1,i=1...\n');
            if usingOctave, fflush(fid); end
          end;
          G1=G(:,:,l); 
       end;
    elseif l>1 && iter==1 && size(m0,2)==1 && reuse_first_G
       if verbose>1
          fprintf(fid,'    Reusing equivalent finite diffs from l=1,i=1...\n');
          if usingOctave, fflush(fid); end
       end;
       G(:,:,l)=G1;
    end
else
    % Compute directly from analytical derivative function:
    if verbose>1
      fprintf(fid,'    Computing derivatives...\n');
      if usingOctave, fflush(fid); end
    end;
    G(:,:,l)=feval(derivfunc,m(:,l,i),varargin{:});
    dpred(:,l,i)=feval(fwdfunc,m(:,l,i),varargin{:});
end


% Prewhiten the local jacobian matrix and residuals:
if verbose>1
   fprintf(fid,'    Prewhitening local jacobian and residuals...\n'); 
   if usingOctave, fflush(fid); end
end;
if size(covd,2)==1
  Ghat=zeros(size(G(:,:,1)));
  for q=1:size(G,1);
     Ghat(q,:)=invsqrtcovd(q)*G(q,:,l); % weighting rows of G, equiv to diag*G
  end
  residshat(:,l,i)=invsqrtcovd.*(dmeas-dpred(:,l,i));
else
  Ghat=invsqrtcovd*G(:,:,l);
  residshat(:,l,i)=invsqrtcovd*(dmeas-dpred(:,l,i));
end
 
% On first of both iterations set the vector of lambdas based on G at m0:
if l==1 && iter==1
   if lambdaIn(1)<0  % ie specifying auto-find lambdas
      if verbose>1
        fprintf(fid,'    Computing lambda points for Lcurve...\n');
        if usingOctave, fflush(fid); end
      end;
      % Automatically computing the lamdas (regularization params) for L-curve:
      % These are computed using the first iteration's derivatives, which are
      % the same for all lambas (note only true for the first iteration) since
      % the first m is the same for all lambdas, equaling m0.

      %Per Christian Hansen approach, goes WAY far out on the Lcurve legs but
      %has restrictions on form of L:
      %lambdamax=max(eig(Ghat'*Ghat))/max(min(eig(L'*L)),16*eps);
      %lambdamin=max(min(eig(Ghat'*Ghat))/max(eig(L'*L)),16*eps);
      %lambda=logspace(log10(lambdamax),log10(lambdamin),nlambdas);
      %disp(num2str(lambda,'%g\n'))

      %Ken Creager approach:
      lambda0=sqrt(max(eig(Ghat'*Ghat))/max(eig(L'*L)));
      % except Creager starts Lcurve (ie max misfit) at lambda0 and reduces from 
      % there, whereas here we have lambda0 sortof roughly in the middle of
      % Lcurve somewhere:
      lambda=logspace(log10(lambda0)+maxlambdaexp,...
                      log10(lambda0)+minlambdaexp, nlambdas);
      if verbose>1
        fprintf(fid,'      Lambda(1)=%g\n',lambda(1));
        if usingOctave, fflush(fid); end
      end
   else
      if verbose>1
        fprintf(fid,'    Using lambda points for Lcurve specified by user...\n');
        if usingOctave, fflush(fid); end
      end;
   end
end


% Compute residual and model norms, and objfn which is based on them, at current 
% lambda & iteration.  In Bayesian case (useMpref==2) normrough doesn't have
% same meaning and not very useful outside of the inversion (eg no Lcurve) but
% it's still used in the objective fn.
normmisfit(l,i)=norm(residshat(:,l,i));
normrough(l,i)=norm(L*(m(:,l,i)));
objfn(l,i)=normmisfit(l,i)+lambda(l)^2*normrough(l,i);



% Check condition number to warn about stability of the inv() in dm()=... below:
if verbose>1
  fprintf(fid,'    Computing model perturbation...\n');
  if usingOctave, fflush(fid); end
end;
c=cond(Ghat'*Ghat+lambda(l)^2*L'*L);
if c>1e15
  if ~warningsoff
  fprintf(fid,'      Warning: cond number of (Ghat''*Ghat+lambda^2*L''*L) is large (%g)\n',c);
  if usingOctave, fflush(fid); end
  end
end
% Compute maximum likelihood model perturbation:
if useMpref
  % Caution! This formulation only valid when:
  %    a.) using 0th-order-Tikh (ridge regr), ie if L==eye(size(L)),
  % or b.) using Bayesian formulation, in which case m0 is the prior model mean
  %        and lambda=1 and L=chol(inv(Cprior)).
  if size(m0,2)>1  % ie diff't m0 for each lambda
    dm(:,l,i)=inv(Ghat'*Ghat+lambda(l)^2*L'*L)*(Ghat'*residshat(:,l,i)-...
                                           lambda(l)^2*L'*L*(m(:,l,i)-m0(:,l)));
  else  % if same m0 for each lambda (and this is the expected case when Bayesian)
    dm(:,l,i)=inv(Ghat'*Ghat+lambda(l)^2*L'*L)*(Ghat'*residshat(:,l,i)-...
                                                lambda(l)^2*L'*L*(m(:,l,i)-m0));
  end
else
  % This option is for frequentist framework with higher-order Tikh regularization.
  % Here regu doesn't care about distance from preferred (initial) model, and
  % there is no restriction on L (whereas above required L=I for useMpref case).
  dm(:,l,i)=inv(Ghat'*Ghat+lambda(l)^2*L'*L)*(Ghat'*residshat(:,l,i)-...
                                              lambda(l)^2*L'*L*m(:,l,i));
end


% Check dm for passing model bounds, ie that step is not too far
if check_bnds
if verbose>1
  fprintf(fid,'    Checking bounds...\n');
  if usingOctave, fflush(fid); end
end;
mtilde=m(:,l,i)+dm(:,l,i);  % candidate model based on current dm() value
outofbnds=sum(mtilde<lb) + sum(mtilde>ub);
if outofbnds>0
    % find out of bounds components
    outlow=mtilde<lb;
    outhigh=mtilde>ub;
    if outbnds_method==0
        % if a component out of bnds, revert to half-step in same direction
        j=0;
        while outofbnds>0 && j<maxstephalves  % yes duplicates things on first round,
                                              % but minimal extra calculation and
                                              % I think easier to understand loop
            j=j+1;
            dm(:,l,i)=dm(:,l,i)/2;
            mtilde=m(:,l,i)+dm(:,l,i);
            outofbnds=sum(mtilde<lb) + sum(mtilde>ub);
        end
        if j>1
            action=['passed bounds so stepsize halved^' num2str(j)];
        else
            action='passed bounds so stepsize halved';
        end
    elseif outbnds_method==1
        % if a component out of bnds, reflect at that bnd
        mtilde(outlow)=lb(outlow)+( lb(outlow)-mtilde(outlow) );
        mtilde(outhigh)=ub(outhigh)-( mtilde(outhigh)-ub(outhigh) );
        action='reflected from bounds';
    elseif outbnds_method==2
        % if a component out of bnds, set it equal to that bnd
        mtilde(outlow)=lb(outlow);    % strange code notation here but it works
        mtilde(outhigh)=ub(outhigh);  % (in both Matlab and Octave)
        action='set to bounds';
    end
    if outbnds_method>0  % update dm given bounds behavior when mtilde altered
        dm(:,l,i)=mtilde-m(:,l,i); 
    end
    if verbose>1
        fprintf(fid,'      %d/%d model params %s in lambda(%d)...\n',outofbnds,nparams,action,l);
        if usingOctave, fflush(fid); end
    end;
else
    if verbose>1
      fprintf(fid,'      All parameter updates within bounds.\n');
      if usingOctave, fflush(fid); end
    end;
end
end


% Check dm for decreasing obj fn, ie once again that step is not too far, after
% having hit "convergence" (presently based on dm)
if check_objfn && converged
if verbose>1
  fprintf(fid,'    Checking for decreasing objfn...\n');
  fprintf(fid,'      Calculating fwdprob for preliminary candidate step...\n');
  if usingOctave, fflush(fid); end
end;
mtilde=m(:,l,i)+dm(:,l,i);  % candidate model based on current dm() value
dpred_tilde=feval(fwdfunc,mtilde,varargin{:});
if size(covd,2)==1
  residshat_tilde=invsqrtcovd.*(dmeas-dpred_tilde);
else
  residshat_tilde=invsqrtcovd*(dmeas-dpred_tilde);
end
normmisfit_tilde=norm(residshat_tilde);
normrough_tilde=norm(L*mtilde);
objfn_tilde=normmisfit_tilde+lambda(l)^2*normrough_tilde;
decreasing_objfn=objfn_tilde<=objfn(l,i);
fprintf(fid,'        objfn(l,i)=%g, objfn_tilde=%g\n',objfn(l,i),objfn_tilde);
if ~decreasing_objfn
    j=0;
    while ~decreasing_objfn && j<maxstephalves
        j=j+1;
        dm(:,l,i)=dm(:,l,i)/2;
        mtilde2=m(:,l,i)+dm(:,l,i);
        if verbose>1
          fprintf(fid,'      Calculating fwdprob for halved candidate step...\n');
          if usingOctave, fflush(fid); end
        end;
        dpred_tilde2=feval(fwdfunc,mtilde2,varargin{:});
        if size(covd,2)==1
          residshat_tilde2=invsqrtcovd.*(dmeas-dpred_tilde2);
        else
          residshat_tilde2=invsqrtcovd*(dmeas-dpred_tilde2);
        end
        normmisfit_tilde2=norm(residshat_tilde2);
        normrough_tilde2=norm(L*mtilde2);
        objfn_tilde2=normmisfit_tilde2+lambda(l)^2*normrough_tilde2;
        fprintf(fid,'        objfn(l,i)=%g, objfn_tilde2=%g, objfn_tilde=%g, j=%d  \n',...
                                            objfn(l,i),objfn_tilde2,objfn_tilde,j);
        if objfn_tilde2 > objfn_tilde
          fprintf(fid,'        WARNING: hump just found between last step and candidate step,\n');
          fprintf(fid,'        implying a region not satisfying monotonicity...\n');
          return;
        else  % objfn_tilde2 <= objfn_tilde
          decreasing_objfn=objfn_tilde2<=objfn(l,i);
        end
        objfn_tilde=objfn_tilde2;
    end
    if verbose>1
        if j>1
            fprintf(fid,'      Non-decreasing objfn initially, so stepsize halved^%d; objfn now decreasing.\n',j);
        else
           fprintf(fid,'      Non-decreasing objfn initially, so stepsize halved; objfn now decreasing.\n');
        end
        if usingOctave, fflush(fid); end
    end;
else
    if verbose>1
      fprintf(fid,'      Parameter update satisfies decreasing objfn.\n');
      if usingOctave, fflush(fid); end
    end;
end
end


% Update the model vector with the new perturbation:
if keepalliters
    m(:,l,i+1)=dm(:,l,i)+m(:,l,i);
    normdm(l,i)=norm(dm(:,l,i));
else
    % It's a bit of a hack, but note that i in here always =1 if keepalliters=0:
    m(:,l,1)=dm(:,l,i)+m(:,l,i);
    normdm(l,iter)=norm(dm(:,l,i));  % i=1 since keepalliters=0, so use iter here.
    % FIXME: hm, the above means that at final iteration, the m(:,l,1) won't
    % technically correspond to the resids & derivs for that i,l.  But if it's
    % really the final iteration such that the solution converged, this shouldn't
    % matter.  By comparison, up in the "keepalliters" part of the if block,
    % at end the m() will have one more element in iters dimension than normdm
    % since the first iter for m() holds the initial m vector.
end


if i>1 && keepalliters
        % (since requires previous iteration's results for comparison in order
        % check for convergence)
        % Note goal is to go one step past convergence, ie obtain m_{i+1},
        % so that can have derivs & resids at soln point m_i, so that's why the
        % resids & Ghat here are from before the model vector update that just 
        % happened.
        % Anyway, model perturbation after convergence should be so small that
        % additional update is negligible.
   if verbose>1
     fprintf(fid,'    Checking convergence criteria for grad(Fobj), dm, & df...\n');
     if usingOctave, fflush(fid); end
   end;
   % for now, "convergence" is based only on dm getting really small:
   % (still analyzing the full convergence test)   FIXME
   %[converged,gradtest,mtest,ftest]=...
   [dummy,gradtest,converged,ftest]=...
   convinfo(fwdfunc,epsr,Ghat,residshat(:,l,i-1),residshat(:,l,i),...
            m(:,l,i-1),m(:,l,i),verbose,fid);
end


if verbose>1
  fprintf(fid,'    Norm(deltam)=%g\n',normdm(l,i));
  fprintf(fid,'    Norm(datamisfit)=%g\n',normmisfit(l,i));
  fprintf(fid,'    Norm(roughness)=%g\n',normrough(l,i));
  fprintf(fid,'    Norm(m)=%g\n',norm(m(:,l,i)));
  if verbose==4
     fprintf(fid,'    dm={');
     for q=1:length(m0)
        fprintf(fid,' %g',dm(q,l,i));
     end
     fprintf(fid,' }\n');
     fprintf(fid,'    m={'); for q=1:length(m0)
        fprintf(fid,' %g',m(q,l,i));
     end
     fprintf(fid,' }\n');
     fprintf(fid,'    1/m={');
     for q=1:length(m0)
        fprintf(fid,' %g',1/m(q,l,i));
     end
     fprintf(fid,' }\n');
  end
  fprintf(fid,'    Objfn=%g\n',objfn(l,i));
  if usingOctave, fflush(fid); end
end

   
if verbose>1
  % output compute time:
  mytime=toc;
  fprintf(fid,'    This iteration took %f seconds of wall-clock time.\n',mytime);
  % output convergence info:
  if converged
    % note mechanism to go check_objfn_iters iters past "convergence" at i_conv(l) iters:
    if i_conv(l)==0, i_conv(l)=iter; end
    fprintf(fid,'  CONVERGED at i=%d based on dm criteria.\n',i_conv(l));
    if iter>=i_conv(l)+check_objfn_iters, break; end  % break out of iters loop at end
    if iter>=i_conv(l)
      fprintf(fid,'  continuing with %d more iters imposing decreasing objfn...\n',...
                                        check_objfn_iters-(iter-i_conv(l)));
    end
  else
    if i_conv(l)>0
      fprintf(fid,'    Reseting convergence flag based on suddenly increasing dm.\n',i_conv(l));
      i_conv(l)=0;  % reset; ie must have 3 converged iters to consider it good
    end
  end
  if usingOctave, fflush(fid); end
end


end;  % end of iteration loop over i

if verbose>1
  if ~converged && iter==maxiters
    fprintf(fid,'  Max iterations reached without convergence.\n');
    if usingOctave, fflush(fid); end
  end
end


% Compute model covariance and resolution matrices for current lambda:
if verbose>1
  fprintf(fid,'  Computing cov and res matrices for Lambda(%d)...\n',l);
  if usingOctave, fflush(fid); end
end;

% Note Ghat corresponds to final model solution m(:,l,i), not m(:,l,i)+dm(:,l,i).
if useMpref<2  % frequentist
  Gginv=inv(Ghat'*Ghat+lambda(l)^2*L'*L)*Ghat';
  C(:,:,l)=Gginv*Gginv';  % model cov matrix for current lambda
  R(:,:,l)=Gginv*Ghat;  % model resolution matrix for current lambda
  % data res matrix D is too big to save whole thing, so just saving its diag:
  Ddiag(:,l)=proddiag(Ghat,Gginv);
else % Bayesian when useMpref==2
  C(:,:,l)=inv(Ghat'*Ghat+lambda(l)^2*L'*L);  % posterior model covariance
                           % note lambda(l) should always be 1 in this case,
                           % and L=chol(inv(Cpriod)) but note notation confusion
                           % in that the "L" does not refer to "left triangular"
                           % but the frequentist findiff operator from earlier!
                           % Sorry but what to do.  "L" here is right triangular.
  % set these empty as they are not relevant for Bayesian formulation:
  R(:,:,l)=[];
  Ddiag(:,l)=[];
end


if saveIntermediate
   if verbose>1
      fprintf(fid,'    saving intermediate workspace to file invGN_%d.mat...\n',pid);
      if usingOctave, fflush(fid); end
   end;
   eval(['save -v7 /tmp/invGN_' num2str(pid) '.mat']);
end


end;  % end of loop over lambdas index l


if verbose
  fprintf(fid,'Finished all iterations for all lambdas.\n');
  if usingOctave, fflush(fid); end
end

