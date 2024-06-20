import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, feature_size):
        super(LSTM,self).__init__()
        '''
        DeepVO LSTM
        '''
        self.lstm1 = nn.LSTM(input_size=feature_size, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(in_features=128, out_features=6)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0)
        self._reinitialize()

        self.fc1 = nn.Linear(in_features=feature_size, out_features=128)
        self.tanh_1 = nn.Tanh()
        # self.leaky_relu_1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.tanh_2 = nn.Tanh()
        # # self.leaky_relu_2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(in_features=128, out_features=6)

        self.rot = nn.Linear(in_features=128, out_features=9)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

        # nn.init.normal_(self.rot.weight, 0, 0.01)
        nn.init.xavier_uniform_(self.rot.weight)
        nn.init.zeros_(self.rot.bias)

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, input):

        x, hc = self.lstm1(input)
        x, hc = self.lstm2(x)
        x = self.dropout(x)
        p = self.out(x)

        x2 = self.fc1(input)
        x2 = self.tanh_1(x2)
        x2 = self.fc2(x2)
        x2 = self.tanh_2(x2)
        r = self.rot(x2)
        
        return p, r

if __name__ == '__main__':
    dummy_input = torch.randn(8, 7, 1024*6*10)
    lstm = LSTM(1024*6*10)
    poses = lstm(dummy_input)
    print(poses.shape)