import torch
import torchvision

        
class TV_loss(torch.nn.Module):
    def __init__(self,TVLoss_weight=0.015):
        super(TV_loss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        
        replicate_padding = torch.nn.ReflectionPad2d(1)
        x = replicate_padding(x)
        
        batch_size = x.size()[0]

        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])

        h_tv = torch.abs(x[:,:,1:,:]-x[:,:,:-1,:]).sum()
        w_tv = torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]).sum()
        
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
