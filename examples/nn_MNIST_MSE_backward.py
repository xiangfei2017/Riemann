# coding: utf-8
import numpy as np
import sys,os,time
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))  # 添加父目录到路径
import riemann as rm

def MSELoss(outputs, targets, reduction:str='mean'):
    mse = (outputs - targets) ** 2.
    if reduction == 'mean':
        ret = rm.mean(mse)
    elif reduction == 'sum':
        ret = rm.sum(mse)    
    return ret

# neural network class definition
class neuralNetwork:   
    # initialise the neural network
    def __init__(self, 
                 inputnodes, hiddennodes, outputnodes,                  
                 hact=rm.nn.functional.relu, oact=rm.nn.functional.sigmoid):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        
        self.wih = rm.normal(0.0,np.power((self.inodes+self.hnodes)/2, -0.5), 
                            (self.inodes,self.hnodes))
        self.wih.requires_grad = True
        
        self.who = rm.normal(0.0, np.power((self.hnodes+self.onodes)/2, -0.5), 
                             (self.hnodes,self.onodes))
        self.who.requires_grad = True

        self.bh = rm.normal(0.0,np.power((self.inodes+self.hnodes)/2,-0.5),
                             (self.hnodes,))
        self.bh.requires_grad = True

        self.bo = rm.normal(0.0,np.power((self.hnodes+self.onodes)/2, -0.5), 
                            (self.onodes,))
        self.bo.requires_grad = True
        
        # learning rate
        self.lr = 0.01
        
        # activation function
        self.hactfun = hact
        self.oactfun = oact
        self.data_list = []

        return
    
    # forward the neural network
    def forward(self, inputs):
        # print(inputs.data.shape, targets.data.shape)
        # 10*784 100*784 100
        hidden_inputs = inputs @ self.wih + self.bh
        hidden_outputs = self.hactfun(hidden_inputs)
        final_inputs = hidden_outputs @ self.who  + self.bo
        final_outputs = self.oactfun(final_inputs)
        
        return final_outputs
      
    def step(self):       
        self.wih.data -= self.lr * self.wih.grad.data
        self.bh.data  -= self.lr * self.bh.grad.data
        self.who.data -= self.lr * self.who.grad.data
        self.bo.data  -= self.lr * self.bo.grad.data

        self.wih.grad = None
        self.bh.grad = None
        self.who.grad = None
        self.bo.grad = None

        return    

    def train(self,dataset,batchsize,lrates):        
        epochs = len(lrates)
        for e in range(epochs):
            batch_inputs = rm.empty(batchsize[e],self.inodes)
            batch_targets = rm.empty((batchsize[e],self.onodes))
            self.lr = lrates[e]

            batch_idx = 0
            for label,input,target in tqdm(dataset,desc='progress'):                
                batch_inputs[batch_idx] = input
                batch_targets[batch_idx] = target
                batch_idx += 1
                
                if batch_idx == batchsize[e] :
                    final_outputs = self.forward(batch_inputs)                    
                    loss = MSELoss(final_outputs, batch_targets)
                    loss.backward()
                    self.step()
                    batch_idx = 0
            pass
        pass
        return

# end of class

def get_file_path(subfolder,filename):
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir / subfolder / filename
    return file_path

def load_training_data(data_file_path):
    # load the mnist training data CSV file into a list
    
    data_list = []
    with open(data_file_path, 'r') as training_data_file:
        for record in tqdm(training_data_file):
            all_values = record.split(',')
            label = int(all_values[0])

            target = rm.full((10,),fill_value=0.01)
            target[label] = 0.99  

            image_numpy = (np.asarray(all_values[1:],dtype=np.float32()) / 255.0 * 0.99) + 0.01
            image = rm.tensor(image_numpy,dtype=rm.get_default_dtype())
            data_list.append((label,image,target))

    return data_list

def test_nn_perf(test_data_file_path):
    # load the mnist test data CSV file into a list
    dataset = load_training_data(test_data_file_path)

    # counts for how well the network performs, initially 0
    counts = 0
    # go through all the records in the test data set
    for correct_label,input,target in tqdm(dataset,desc='progress'): 
        output = n.forward(input)
        label = output.argmax()
        if (label == correct_label):
            counts += 1
    pass

    return counts / len(dataset)

# rm.set_default_dtype(rm.float64)

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes)


def clear_screen():
    """跨平台清屏函数（自动识别系统）"""
    if os.name == 'nt':
        os.system('cls')  # Windows系统指令
    else:
        os.system('clear')  # Linux/MacOS指令
    return

clear_screen()

print("loading training data...")
starttime = time.time()
dataset = load_training_data(get_file_path("mnist_dataset","mnist_train.csv"))
endtime = time.time()
print(f"loading seconds:{endtime-starttime:.2f}")

# train the neural network
# batchsize = [100,100,100]
# learning_rates = [0.01,0.01,0.01]
batchsize = [1]
learning_rates = [0.01]

print("start training...")
print(f'samples: {len(dataset)}')
print(f'train batch size: {batchsize}')
print(f'learning rates: {learning_rates}')
print(f'train epochs: {len(learning_rates)}')

starttime = time.time()
n.train(dataset,batchsize,learning_rates)
endtime = time.time()
print(f"train seconds:{endtime-starttime:.2f}")

# test the neural network
print("start testing...")
starttime = time.time()
perf = test_nn_perf(get_file_path("mnist_dataset","mnist_test.csv"))
endtime = time.time()
print(f"test seconds:{endtime-starttime:.2f}")
print (f"performance = {perf:.2f}")

