function Newton_method_regularized(train_X,train_Y,test_X,test_Y,w_0,b_0)

% calculate w and b
train_X_new=[ones(size(train_X,1),1),train_X];
test_X_new=[ones(size(test_X,1),1),test_X];
[train_X_n,train_X_f]=size(train_X_new);
[test_X_n,test_X_f]=size(test_X_new);

T=50;
e=exp(1);
lambda=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];

train_re_CE=zeros(T,length(lambda));     %regularized corss-entropy error
test_re_CE=zeros(T,length(lambda));     %regularized corss-entropy error
train_l2_norm=zeros(1,length(lambda));

tic
% regularized model
for l=1:length(lambda)
    w_new=[b_0;w_0];
    w=[w_new,zeros(length(w_new),T)];
    
    for t=1:T
        sum_w_up=zeros(train_X_f,1);
        sum_w_down=zeros(test_X_f,test_X_f);
  
        for n=1:train_X_n
            
            sigmoid=1/(1+e^(-(w(:,t)'*train_X_new(n,:)')));
            if sigmoid<10^(-16)
                sigmoid=10^(-16);
            elseif sigmoid>(1-10^(-16))
                sigmoid=1-10^(-16);
            end
            
            sum_w_up=sum_w_up+(sigmoid-train_Y(n))*train_X_new(n,:)';
            sum_w_down=sum_w_down+train_X_new(n,:)'*train_X_new(n,:)*sigmoid*(1-sigmoid);
        end
        matrix_lambda=[0;ones(test_X_f-1,1)];
        w(:,t+1)=w(:,t)-pinv(sum_w_down+2*lambda(l)*diag(matrix_lambda))*(sum_w_up+2*lambda(l)*diag(matrix_lambda)*w(:,t));
        
      
        % CE for train data
        for n=1:train_X_n
            
            sigmoid=1/(1+e^(-(w(:,t+1)'*train_X_new(n,:)')));
            if sigmoid<10^(-16)
                sigmoid=10^(-16);
            elseif sigmoid>(1-10^(-16))
                sigmoid=1-10^(-16);
            end
            train_re_CE(t,l)=train_re_CE(t,l)-train_Y(n)*log(sigmoid)-(1-train_Y(n))*log(1-sigmoid);
            
        end
        
        train_re_CE(t,l)=train_re_CE(t,l)+lambda(l)*norm(w(:,t+1))^2;
       
        
        % CE for test data
        for n=1:test_X_n
            
            sigmoid=1/(1+e^(-(w(:,t+1)'*test_X_new(n,:)')));
            if sigmoid<10^(-16)
                sigmoid=10^(-16);
            elseif sigmoid>(1-10^(-16))
                sigmoid=1-10^(-16);
            end
            test_re_CE(t,l)=test_re_CE(t,l)-test_Y(n)*log(sigmoid)-(1-test_Y(n))*log(1-sigmoid);
            
        end
        test_re_CE(t,l)=test_re_CE(t,l)+lambda(l)*norm(w(:,t+1))^2;
    end
     train_l2_norm(l)=norm(w(:,51));
end
% plot cross-entropy function value
toc

x=1:50;
plot(x,train_re_CE(:,1),'-.b',x,train_re_CE(:,2),'or-',x,train_re_CE(:,3),'gx-',x,train_re_CE(:,4),'c+-',x,train_re_CE(:,5),'m*-',x,train_re_CE(:,6),'yv-',x,train_re_CE(:,7),'kd-',x,train_re_CE(:,8),'gd-',x,train_re_CE(:,9),'bs-',x,train_re_CE(:,10),'g<-',x,train_re_CE(:,11),'r>-')

title('Regularized Cross-Entropy value');
xlabel('T');
ylabel('Cross-Entropy value');
legend('0','0.05','0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5');

for l=1:length(lambda)
train_l2_norm(l)
end
test_re_CE(50,:)
