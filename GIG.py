import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle 
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.decomposition import PCA
#np.set_printoptions(threshold=np.inf) #全部输出 
pca = PCA(n_components=5)


df = pd.read_excel('data.xlsx')
train = df




X_train = train.iloc[:,1:]
X_train = X_train.drop('AA',axis=1)


def cleandata(X_train):
    i = 1
    for i in range (124):
        X_train.loc[i,'G'] = ((X_train.loc[i,'G'])+(X_train.loc[i,'H']))/2
        X_train.loc[i,'I'] = ((X_train.loc[i,'I'])+(X_train.loc[i,'J']))/2
        X_train.loc[i,'K'] = ((X_train.loc[i,'K'])+(X_train.loc[i,'L']))/2
        X_train.loc[i,'M'] = ((X_train.loc[i,'M'])+(X_train.loc[i,'N']))/2
        X_train.loc[i,'O'] = ((X_train.loc[i,'O'])+(X_train.loc[i,'P']))/2
        X_train.loc[i,'Q'] = ((X_train.loc[i,'Q'])+(X_train.loc[i,'R']))/2
        X_train.loc[i,'S'] = ((X_train.loc[i,'S'])+(X_train.loc[i,'T']))/2
        X_train.loc[i,'U'] = ((X_train.loc[i,'U'])+((X_train.loc[i,'V'])/100))/2
        X_train.loc[i,'W'] = ((X_train.loc[i,'W'])+(X_train.loc[i,'X']))/2
        X_train.loc[i,'Y'] = ((X_train.loc[i,'Y'])+(X_train.loc[i,'Z']))/2
        X_train.loc[i,'AC'] = ((X_train.loc[i,'AC'])+(X_train.loc[i,'AD']))/2
        X_train.loc[i,'AE'] = ((X_train.loc[i,'AE'])+(X_train.loc[i,'AF']))/2
        X_train.loc[i,'AG'] = ((X_train.loc[i,'AG'])+(X_train.loc[i,'AH']))/2
    X_train.drop(['H','I','J','K','L','M','N','O','P','Q','R','T','V','X','Z','AD','AF','AH','F','G','S','U','AB','AC','AG','E','Y'], axis=1, inplace=True) 

            #print (i)

cleandata(X_train)
pca.fit(X_train)
X_train = pca.fit_transform(X_train)
#print(X_train) 
#sns.boxplot(data=X_train)
#print(X_train)
"""
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 14))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True,
        annot=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(X_train)
"""
Y_train = train.iloc[:,0]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.astype(float)

Y_sum = float(Y_train.max()+Y_train.min())


for i in range(5):
    X_train[:,i] = X_train[:,i] / (X_train[:,i].max()+X_train[:,i].min())

"""  
for j in range(124):
    Y_train[j] = Y_train[j] / Y_sum
"""  

x = tf.placeholder(tf.float32,[None,5],name="X")
y = tf.placeholder(tf.float32,[None,1],name="Y")


with tf.name_scope("Model"):
    w1 = tf.Variable(tf.random_normal([5,1],stddev=0.01,name="W1"))
    w2 = tf.Variable(tf.random_normal([5,1],stddev=0.01,name="W2"))
    w3 = tf.Variable(tf.random_normal([5,1],stddev=0.01,name="W3"))
    b = tf.Variable(1.0,name="b")
    def model(put,weight1,weight2,weight3,offset):
        return tf.matmul(put**3,weight3) + tf.matmul(put**2,weight1) + tf.matmul(put,weight2) + offset
"""  
with tf.name_scope("Model"):
    w1 = tf.Variable(tf.random_normal([7,1],stddev=0.01,name="W1"))
    #w2 = tf.Variable(tf.random_normal([7,1],stddev=0.01,name="W2"))
   # w3 = tf.Variable(tf.random_normal([7,1],stddev=0.01,name="W3"))
    b = tf.Variable(1.0,name="b")
    def model(put,weight1,weight2,weight3,offset):
"""
pred = model(x,w1,w2,w3,b)

train_epochs = 200
learning_rate = 0.001
with tf.name_scope("Lossfunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs,ys in zip(X_train,Y_train):
        xs = xs.reshape(1,5)
        ys = ys.reshape(1,1)
    _, loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
    loss_sum = loss_sum + loss
    x_value,y_value = shuffle(X_train,Y_train)
    
    b0temp = b.eval(session=sess)
    w1temp = w1.eval(session=sess)
    w2temp = w2.eval(session=sess)
    w3temp = w3.eval(session=sess)

    loss_average = loss_sum/len(Y_train)

    #print("epoch:",epoch+1,"loss:",loss_average,"b:",b0temp,"w1:",w1temp,"w2:",w2temp,"w3:",w3temp)
    



test = df

X_test = test.iloc[:,1:]
X_test= X_test.drop('AA',axis=1)
Y_test = test.iloc[:,0]
Y_validation = np.array(Y_test)
#Y_validation = Y_validation.astype(float)
cleandata(X_test)

X_Arrtest = np.array(X_test)

#pca.fit(X_Arrtest)
X_Arrtest = pca.fit_transform(X_Arrtest)

for i in range(5):
    X_Arrtest[:,i] = X_Arrtest[:,i] / (X_Arrtest[:,i].max()+X_Arrtest[:,i].min())

"""  
for i in range(124):
    Y_validation[i] = Y_validation[i] / (Y_validation.max()+Y_validation.min())
"""  
predict = sess.run(pred,feed_dict={x:X_Arrtest})



n = np.arange(124)
n = n.reshape(124,1)

a = []
b = []
c = []
for i in range(124):
    if abs(predict[i] - Y_validation[i]) <= 30:
       #print("测试输入：",X_Arrtest[i])
       #print("预测值：",predict[i])
       #print("验证：",Y_validation[i])
       a.append(predict[i]) 
       b.append(Y_validation[i])
       c.append(i)
#print(np.array(a).shape)
#print(np.array(b).reshape(35,1).shape)
#print(np.array(c).reshape(35,1).shape)     
a = np.array(a)
b = np.array(b).reshape(35,1)
c = np.array(c).reshape(35,1)
C = np.arange(1,36)
plt.plot(C,a)
plt.plot(C,b)     
plt.show()  
"""  
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(n,predict)
plt.plot(n,Y_validation)
plt.sca(ax1)
plt.show()
"""