from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import glob,os
import torch


device='cuda' if torch.cuda.is_available() else 'cpu'
class Linear_QNet(nn.Module) :
    def __init__(self,input_size,hidden_size,output_size) -> None:
        super(Linear_QNet,self).__init__()
        self.l0=nn.Linear(input_size,hidden_size)
        self.l1=nn.Linear(hidden_size,output_size)


    def forward(self,x):
        x=self.l0(x)
        x=self.l1(x)
     
       
        return x
    
    def save(self,file_name='model.pth') :
        model_fpath="./model"
        if not os.path.exists(model_fpath) :
            os.makedirs(model_fpath)

        file_name=os.path.join(model_fpath,file_name)
        torch.save(self.state_dict(),file_name)



class QTrainer :
    def __init__(self,model,lr,gamma) -> None:
        self.lr=lr 
        self.gamma=gamma
        self.model=model 
        self.optimizer=optim.Adam(model.parameters(),self.lr)
        self.criterion=nn.MSELoss()

    def train_step(self,state,action,reward, next_state, done ):
        state=torch.tensor(state,dtype=torch.float).to(device)
        next_state=torch.tensor(next_state,dtype=torch.float).to(device)
        action=torch.tensor(action,dtype=torch.float).to(device)
        reward=torch.tensor(reward,dtype=torch.float).to(device)

        if len(state.shape)==1 :
            state=torch.unsqueeze(state,0).to(device)
            next_state=torch.unsqueeze(next_state,0).to(device)
            action=torch.unsqueeze(action,0).to(device)
            reward=torch.unsqueeze(reward,0).to(device)
            done=(done,)
        

        ## Predicted Q value 
        pred=self.model(state)

        #Bellman equation 
        target=pred.clone()
        for idx in range(len(done)) :
            Q_new=reward[idx]
            if not done : 
                Q_new=reward[idx]+self.gamma*torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()]=Q_new
        

        self.optimizer.zero_grad()
        loss=self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()




        










