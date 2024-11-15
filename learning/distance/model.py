import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image

#VERY basic model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)
        self.act3 = nn.ReLU()
        self.lin1 = nn.Linear(flat_size, 512)
        self.act_output = nn.Sigmoid()
        
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.act_output(self.lin1(x))        
        return x


# Very basic train not tested
def train(model):
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100
    batch_size = 10
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

if __name__ == "__main__":
    print("elo")
    dist = -0.0033901180039446403
    im = Image.open("example.png")
    model = CNNModel()
    print(model)