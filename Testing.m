
%tic;
pnew =dataTesting';
pnewn = mapstd('apply',pnew,ps1);
%pnewtrans = processpca('apply',pnewn,ps2);
out= sim(net,pnewn);

% anewn = sim(net,pnewn);
% anew = mapminmax('reverse',anewn,ts);
% at=anew';

%save output;
%toc;
%gensim(net,.001);