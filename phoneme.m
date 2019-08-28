load phoneme
p = con2seq(dataTraining');
t = con2seq(train_target');
lrn_net = layrecnet(1,10);
lrn_net.trainParam.goal=0.001;
lrn_net.trainFcn = 'trainlm';
lrn_net.trainParam.show = 5;
lrn_net.trainParam.epochs = 10000000;
lrn_net = train(lrn_net,p,t);