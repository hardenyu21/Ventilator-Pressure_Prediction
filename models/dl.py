import torch
from torch import nn


class MLP_tabular(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.n_features = 39 if args.is_feature_eng else 5
        self.input = nn.Linear(self.n_features, 64)
        
        self.hidden = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                                    nn.Linear(256, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 64))
        
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
    
        return x

class MLP_Sequence(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_features = 39 * 80 if args.is_feature_eng else 5 * 80
        self.flatten = nn.Flatten()
        self.input = nn.Linear(self.n_features, 4096)
        self.hidden = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(),
                                    nn.Linear(2048, 2048), nn.ReLU(),
                                    nn.Linear(2048, 1024), nn.ReLU(),
                                    nn.Linear(1024, 512))
        self.output = nn.Linear(512, 80)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
    
        return x
    
class CNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.n_features = 39  if args.is_feature_eng else 5
        self.layer1 = nn.Sequential(nn.Linear(self.n_features, 80), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 4, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(4), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels = 4, out_channels = 16, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(16), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 64, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(64), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 256, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(256), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer6 = nn.Sequential(nn.Flatten(), nn.Linear(6400, 2048),
                                    nn.ReLU(), nn.Linear(2048, 512),
                                    nn.ReLU(), nn.Linear(512, 80))
        

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
    
        return x

class CNN_Residual(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.n_features = 39  if args.is_feature_eng else 5
        self.layer1 = nn.Sequential(nn.Linear(self.n_features, 80), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 4, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(4),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels = 4, out_channels = 16, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(16),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 64, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 256, 
                                              kernel_size = 5, padding = 2),
                                    nn.BatchNorm2d(256),
                                    nn.MaxPool2d(kernel_size = 5, padding = 2, stride = 2))
        self.layer6 = nn.Sequential(nn.Flatten(), nn.Linear(6400, 2048),
                                    nn.ReLU(), nn.Linear(2048, 512),
                                    nn.ReLU(), nn.Linear(512, 80))
        
        self.connect1 = nn.Conv2d(in_channels = 1, out_channels = 4, 
                                  kernel_size = 1, stride = 2)
        self.connect2 = nn.Conv2d(in_channels = 4, out_channels = 16, 
                                  kernel_size = 1, stride = 2)
        self.connect3 = nn.Conv2d(in_channels = 16, out_channels = 64, 
                                  kernel_size = 1, stride = 2)
        self.connect4 = nn.Conv2d(in_channels = 64, out_channels = 256, 
                                  kernel_size = 1, stride = 2)
        
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(self.layer2(x) + self.connect1(x))
        x = self.relu(self.layer3(x) + self.connect2(x))
        x = self.relu(self.layer4(x) + self.connect3(x))
        x = self.relu(self.layer5(x) + self.connect4(x))
        x = self.layer6(x)

        return x  
    

class RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size = 39 if args.is_feature_eng else 5
        hidden_size = 256
        num_layers = 5
        if args.model_name == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, 
                              num_layers, batch_first = True)
        if args.model_name == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, 
                               num_layers, batch_first = True)
        if args.model_name == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 
                              num_layers, batch_first = True)
            
        self.fc = nn.Linear(hidden_size, 1)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        out, _ = self.rnn(x, None)
        out = self.fc(out).squeeze(-1)
        return out

def xavier_init(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_normal_(layer.weight)


def get_mlp(args):
    if args.model_name  == 'mlp_tabular':
        return MLP_tabular(args).apply(xavier_init)
    else:
        return MLP_Sequence(args).apply(xavier_init)

def get_cnn(args):
    if args.model_name  == 'cnn':
        return CNN(args).apply(xavier_init)
    if args.model_name == 'cnn_residual':
        return CNN_Residual(args).apply(xavier_init)

def get_rnn(args):
    return RNN(args)