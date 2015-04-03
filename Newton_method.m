function Newton_method(train_X,train_Y,test_X,test_Y,w_0,b_0)

% calculate w and b
train_X_new=[ones(size(train_X,1),1),train_X];
test_X_new=[ones(size(test_X,1),1),test_X];
[train_X_n,train_X_f]=size(train_X_new);
[test_X_n,test_X_f]=size(test_X_new);

T=50;
train_un_CE=zeros(T,1);     %unregularized corss-entropy error
test_un_CE=zeros(T,1);
train_l2_norm=zeros(1,5);

e=exp(1);

% unregularized model
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
    
    w(:,t+1)=w(:,t)-pinv(sum_w_down)*sum_w_up;
    
    % CE for train data
    for n=1:train_X_n
        
        sigmoid=1/(1+e^(-(w(:,t+1)'*train_X_new(n,:)')));
        if sigmoid<10^(-16)
            sigmoid=10^(-16);
        elseif sigmoid>(1-10^(-16))
            sigmoid=1-10^(-16);
        end
        train_un_CE(t,1)=train_un_CE(t,1)-train_Y(n)*log(sigmoid)-(1-train_Y(n))*log(1-sigmoid);
        
    end
    
    train_l2_norm=norm(w(:,51));
    
    % CE for test data
    for n=1:test_X_n
        
        sigmoid=1/(1+e^(-(w(:,t+1)'*test_X_new(n,:)')));
        if sigmoid<10^(-16)
            sigmoid=10^(-16);
        elseif sigmoid>(1-10^(-16))
            sigmoid=1-10^(-16);
        end
        test_un_CE(t,1)=test_un_CE(t,1)-test_Y(n)*log(sigmoid)-(1-test_Y(n))*log(1-sigmoid);
        
    end
    
end

% plot cross-entropy function value
x=1:50;
figure
plot(x,train_un_CE(:,1),'-.b')

title('Unregularized Cross-Entropy value');
xlabel('T');
ylabel('Cross-Entropy value');
legend('train');

train_l2_norm
test_un_CE(50,1)
