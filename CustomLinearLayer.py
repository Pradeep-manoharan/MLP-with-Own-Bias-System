import torch
import torch.nn as nn


class CustomLinearLayer(nn.Module):
    def __init__(self, input_size, output_size,batch_size=100):
        super(CustomLinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(input_size,output_size))  # 500,784
        self.bias = nn.Parameter(torch.Tensor(batch_size,input_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform element-wise multiplication without summing up
        # bias = self.bias
        # weight = self.weight
        # xu = x.unsqueeze(2)
        # wu = self.weight.unsqueeze(0)
        #result = x.unsqueeze(2) * self.weight.unsqueeze(0)

        #bias_unsqueeze = self.bias.unsqueeze(0)
        # Add the bias matrix
        #shape_result =result.shape
        #result += self.bias
        #result = torch.transpose(result,result.shape[2],result.shape[1])
        #result = torch.mean(result, dim=1)
        #shape = result.shape


        # m(x+b)

        bias = self.bias.shape
        weight = self.weight.shape
        result = x+self.bias
        out = torch.matmul(result,self.weight)

        return out
