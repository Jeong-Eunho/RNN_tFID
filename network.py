import torch

# rRNN architecture class
class rRNN_encoder(torch.nn.Module):
    def __init__(self, hiddenNum, layerNum):
        super().__init__()
        self.gru = torch.nn.GRU(2, hiddenNum, layerNum, batch_first=True)
    def forward(self, x):
        _, hidden_summary = self.gru(x)
        return hidden_summary

class rRNN_decoder(torch.nn.Module):
    def __init__(self, hiddenNum, layerNum):
        super().__init__()
        self.gru     = torch.nn.GRU(2, hiddenNum, layerNum, batch_first=True)
        self.fc_out  = torch.nn.Linear(hiddenNum, 2)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, input, hidden_pre):
        out, hidden_decoder = self.gru(input, hidden_pre)
        out                 = self.dropout(out)
        out                 = self.fc_out(out.squeeze(1))
        fid                 = out.unsqueeze(1)
        return fid, hidden_decoder

class rRNN(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
    def forward(self, x, y, xLen, yLen, teachForce, device):
        hidden = self.encoder(x)
        input  = x[:,-1,:].unsqueeze(1)
        fids   = torch.zeros(x.size(0), yLen - xLen, 2).to(device)
        for t in range(yLen - xLen):
            fid, hidden = self.decoder(input, hidden)
            fids[:,t,:] = fid.squeeze(1)
            input       = fid if (torch.rand(1) > teachForce) else y[:,t,:].unsqueeze(1)
        return fids
    
# cRNN architecture class
class cRNN_encoder(torch.nn.Module):
    def __init__(self, hiddenNum, layerNum):
        super().__init__()
        self.gru = torch.nn.GRU(2, hiddenNum, layerNum, batch_first=True)
    def forward(self, x):
        _, hidden_summary = self.gru(x)
        return hidden_summary

class cRNN_decoder(torch.nn.Module):
    def __init__(self, hiddenNum, layerNum):
        super().__init__()
        self.gru     = torch.nn.GRU(2, hiddenNum, layerNum, batch_first=True)
        self.fc_out  = torch.nn.Linear(hiddenNum, 2)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, input, hidden_pre):
        out, hidden_decoder = self.gru(input, hidden_pre)
        out                 = self.dropout(out)
        out                 = self.fc_out(out.squeeze(1))
        fid                 = out.unsqueeze(1)
        return fid, hidden_decoder

class MappingFirstPoint(torch.nn.Module):
    def __init__(self,xLen):
        super().__init__()
        self.fc_1      = torch.nn.Linear(xLen*2, xLen)
        self.dropout_1 = torch.nn.Dropout(0.5)
        self.relu      = torch.nn.ReLU()
        self.fc_2      = torch.nn.Linear(xLen  , 2)
    def forward(self, input):
        out            = self.fc_1(torch.reshape(input,(input.size(0),-1)))
        out            = self.dropout_1(out)
        out            = self.relu(out)
        first_fid      = self.fc_2(out).unsqueeze(1)
        return first_fid

class cRNN(torch.nn.Module):
    def __init__(self, encoder, decoder, map_first):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.map_first  = map_first
    def forward(self, x, y, xLen, teachForce, device):
        hidden      = self.encoder(x)
        input       = self.map_first(x) if (torch.rand(1) > teachForce) else y[:,0,:].unsqueeze(1)
        fids        = torch.zeros(x.size(0), xLen, 2).to(device)
        fids[:,0,:] = input.squeeze(1)
        for t in range(xLen-1):
            fid, hidden   = self.decoder(input, hidden)
            fids[:,t+1,:] = fid.squeeze(1)
            input         = fid if (torch.rand(1) > teachForce) else y[:,t+1,:].unsqueeze(1)
        return fids