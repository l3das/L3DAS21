import torch
from torch import nn
from torch import randn
from torch import optim
from torch import reshape
from torch import stack
from torch import tensor
import numpy as np
from L3DAS.data import Dataset
from L3DAS.audio_processing import fft_set

class GinNet(nn.Module):
    def __init__(self, n_frames, lstm_h, num_classes, num_locations):
        
        super(GinNet, self).__init__()
        self.n_frames=n_frames
        self.lstm_h=lstm_h

        self.conv1=nn.Conv2d(n_frames,n_frames,(10,1))
        self.conv2=nn.Conv2d(n_frames,n_frames,(10,1))
        self.conv3=nn.Conv2d(n_frames,n_frames,(10,1))

        self.gru=nn.GRU(17424,lstm_h,num_layers=1,bidirectional=False,batch_first=True)
        self.dense1_y1 = nn.Linear(lstm_h, num_classes)
        self.dense1_y2 = nn.Linear(lstm_h, num_locations)

        self.last_sigmoid = nn.Sigmoid()
        self.last_tanh = nn.Tanh()

    def forward(self, x):
        out_conv1=self.conv1(x)
        out_conv2=self.conv2(out_conv1)
        out_conv3=self.conv3(out_conv2)

        gru_input=reshape(out_conv3,(out_conv3.shape[0],out_conv3.shape[1],out_conv3.shape[2]*out_conv3.shape[3]))
        
        gru_out, _ =self.gru(gru_input)

        time_distributed_batch_y1 = []
        time_distributed_batch_y2 = []
        time_distributed_sample_y1 = []
        time_distributed_sample_y2 = []
        for i in range(gru_out.shape[0]):

            for j in range(self.n_frames):
                time_distributed_sample_y1.append(self.dense1_y1(gru_out[i][j]))
                time_distributed_sample_y2.append(self.dense1_y2(gru_out[i][j]))
            
            time_distributed_sample_y1=stack(time_distributed_sample_y1)
            time_distributed_sample_y2=stack(time_distributed_sample_y2)

            time_distributed_batch_y1.append(time_distributed_sample_y1)
            time_distributed_batch_y2.append(time_distributed_sample_y2)

            time_distributed_sample_y1=[]
            time_distributed_sample_y2=[]

        time_distributed_batch_y1=stack(time_distributed_batch_y1)
        time_distributed_batch_y1=self.last_sigmoid(time_distributed_batch_y1)

        time_distributed_batch_y2=stack(time_distributed_batch_y2)
        time_distributed_batch_y2=self.last_tanh(time_distributed_batch_y2)*10

        return time_distributed_batch_y1, time_distributed_batch_y2


num_samples=30
dataset = Dataset('Task2',num_samples=num_samples,frame_len=0.2,set_type='train',mic='A',domain='freq', spectrum='s')
audio, classes, location = dataset.get_dataset()

del dataset
print(end='\n\n')
audio=tensor(audio)
classes=tensor(classes)
location=tensor(location)

step=1
batch_size=3
epochs=100
n_frames=audio.shape[1]
lstm_h=100
num_classes=classes.shape[-1]
num_locations=location.shape[-1]

model = GinNet(n_frames, lstm_h, num_classes,num_locations)

bc = nn.BCELoss()
mse = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=1e-2)
current_epoch = 1
mark_batch=0

while(True):
    batch=audio[mark_batch:mark_batch+batch_size]
    y1_pred, y2_pred = model(batch.float())
    loss_y1 = bc(y1_pred, classes[mark_batch:mark_batch+batch_size].float()) 
    loss_y2 = mse(y2_pred, location[mark_batch:mark_batch+batch_size].float())
    loss = loss_y1 + loss_y2
    print("Epoch: %2d   Timestep: %3d   Classification Loss: %5.5f   Localization Loss: %5.5f   Total Loss: %5.5f" %(current_epoch, step,loss_y1,loss_y2,loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    mark_batch+=batch_size
    step+=1
    if mark_batch>=num_samples:
        mark_batch=0
        current_epoch+=1
        step=1
        print()
    if current_epoch>epochs:
        break
torch.save(model.state_dict(), 'model.pt')