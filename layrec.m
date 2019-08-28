%[X,T] = simpleseries_dataset;
P = con2seq(dataTraining');
T = con2seq(train_target');

net = layrecnet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,P,T);
net = train(net,Xs,Ts,Xi,Ai);
Y =net(Xs,Xi,Ai);
perf=perform(net,Y,Ts)