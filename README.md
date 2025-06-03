# tesla-stock-prediction

I had to perform and predict the price of TESLA stock with the use of LSTM Neural Network which is an RNN architecture used in deep learning as well as for machine learning.

LSTM models give us the freedom to decide what information will be stored and what will be discarded. They are used to store a particular information with period of time.
LSTMs have proved to be highly effective for solving sequence prediction problems.
Since LSTM stores past important information and forgets the unimportant, they are so effective.

## STEP 1: Data Understanding and Manipulation

Eliminated redundant variables and checked if there is any null value or not which are some parts of preliminary steps. Also, I had to check the number of days in the dataset so that I can plot graphs for the further steps. 

## STEP 2: Data Visualisation

Created a simple graph of changes in stock prices of TESLA from the above given data set taken

![image](https://github.com/user-attachments/assets/0eefa496-493b-460e-a14e-61b7c0979fa1)

## STEP 3: Preparing the train and test data

```
X_train  = np.array(X[:200])
y_train = np.array(y[:200])

X_test = np.array(X[200:])
y_test = np.array(y[200:])

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))
```

## STEP 4: Building the LSTM model

First, I defined the hyper-parameters used in the network. Next, I had to input weights for input gate, output gate, forgot gate and memory cell. 

Now, for each point in the window, we will be feeding into the model to get the next output by creating an iteration. The last output is considered and is used to get the prediction for the data set. The working is mentioned below:

```
outputs = []
for i in range(batch_size): 
    
    batch_state = np.zeros([1, hidden_layer], dtype=np.float32) 
    batch_output = np.zeros([1, hidden_layer], dtype=np.float32)
    
    for ii in range(window_size):
        batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][ii], (-1, 1)), batch_state, batch_output)
  
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
```

Now, to calculate the loss, we use the squared difference between the true and predicted values. It takes around 3-5 minutes for the following code to run, so don't worry, and we will get the following output - 

```
Epoch 0/200  Current loss: 0.1782066524028778
Epoch 30/200  Current loss: 0.025821495801210403
Epoch 60/200  Current loss: 0.019261199980974197
Epoch 90/200  Current loss: 0.018246600404381752
Epoch 120/200  Current loss: 0.026575682684779167
Epoch 150/200  Current loss: 0.02041161060333252
Epoch 180/200  Current loss: 0.01604599691927433
```

## STEP 5: Visualise predicted vs actual stock values

![image](https://github.com/user-attachments/assets/9a9c68fb-5840-4c53-bec1-dad508788c25)

You might or might not get a 1-d array error but it won't matter. The graph will be plotted anyways.
From the graph given above, it can be clearly seen that there is a sudden huge rise in the TESLA stock in around last 50 days and that the model that was created, performed good.



