import torch
import numpy as np

hidden_tensor_dim = 2 # dimension 2 is the hidden dimension of the transformer for huggingface

class ReversibleBlock(torch.nn.Module):
    def __init__(self, F, G):
        super(ReversibleBlock, self).__init__()          
        self.G = G
        self.F = F        

    def forward(self, inputs):
        x, attention_mask = inputs
        args = [x, attention_mask, self.F, self.G] + [w for w in self.F.parameters()] + [w for w in self.G.parameters()]        
        debug = False
        if debug: # no memory saving, used for debugging.    
            x1, x2 = torch.chunk(x, 2, dim=hidden_tensor_dim)
            # TODO, all of the contiguous calls seem unneccesary here (for correctness), does calling contiguous manually have any speed improvements?
            # x1, x2 = x1.contiguous(), x2.contiguous()
            Fd = self.F.forward(x2, attention_mask)
            y1 = x1 + Fd
            Gd = self.G.forward(y1)
            y2 = x2 + Gd
            out = torch.cat([y1, y2], dim=hidden_tensor_dim)
        else:            
            out = ReversibleBlockFunction.apply(*args)

        # TODO, check this and the other storage things (the detach() and del below). It seems like this direct setting of resize to 0 does clear memory, but does it add cost? Doesn't seem like it adds costs. But, the `del` and set_() don't seem to save any memory. Is that a problem with the nvidia-smi just not showing it?
        x.storage().resize_(0) 
        return out
     
class ReversibleBlockFunction(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x, attention_mask, F, G, *weights):
        """
        Goes forward through the reversible layer.

        Parameters
        ----------
        ctx : torch.autograd.function.RevNetFunctionBackward
            The backward pass context object        
        *weights : TorchTensor
            weights for F and G in that order {F_w1, ... F_wn, G_w1, ... G_wn}
        """
        # assert it is possible to partition into two equally sized halves        
        assert(x.shape[hidden_tensor_dim] % 2 == 0)

        # store F and G functions in context, accessed in reverse()        
        # TODO, doesn't this mean we are storing memory for the weights of F on GPU, and then we are also storing them in ctx (which is needed in reverse)? Or do they point to the same place.
        ctx.F = F 
        ctx.G = G
        ctx.attention_mask = attention_mask        

        with torch.no_grad():
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=hidden_tensor_dim) # does not duplicate memory, very fast.            
            # x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            # TODO, can we do forwards of F and G in parallel? Will pytorch do that automatically?
            Fr = F.forward(x2, attention_mask) 
            y1 = x1 + Fr
            
            Gr = G.forward(y1)
            y2 = x2 + Gr

            output = torch.cat([y1, y2], dim=hidden_tensor_dim) # TODO, so cat does deepcopy here, is this slow, does compiler fix this automatically? akak, we don't need to actually store y1 and y2, it should put them in the same spot next to eachother in mem.

            # TODO, is the below necessary? Does it do anything?            
            x1.set_()
            x2.set_()
            y1.set_()
            y2.set_()
            Fr.set_()
            Gr.set_()
            del x1, x2, y1, y2, Fr, Gr

        # save the (empty) input and (non-empty) output variables
        ctx.save_for_backward(x.data, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):        
        assert(grad_output.shape[hidden_tensor_dim] % 2 == 0)

        F, G, attention_mask = ctx.F, ctx.G, ctx.attention_mask
        x, output = ctx.saved_tensors

        with torch.no_grad():
            y1, y2 = torch.chunk(output, 2, dim=hidden_tensor_dim)
            # y1, y2 = y1.contiguous(), y2.contiguous()
            
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=hidden_tensor_dim)
            # y1_grad, y2_grad = y1_grad.contiguous(), y2_grad.contiguous()

            # TODO, even though output shouldn't have been saved in the first place, we can destroy it here right?

        # Recreate computation graphs for functions G and F with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        # For more info, see Algorithm 1 of the reversible nets paper https://papers.nips.cc/paper/6816-the-reversible-residual-network-backpropagation-without-storing-activations.pdf
        with torch.set_grad_enabled(True):
            z1_stop = y1.detach()
            z1_stop.requires_grad = True
            G_z1 = G.forward(z1_stop)
            x2 = y2 - G_z1

            x2_stop = x2.detach()
            x2_stop.requires_grad = True
            F_x2 = F.forward(x2_stop, attention_mask)
            x1 = y1 - F_x2

            x1_stop = x1.detach()
            x1_stop.requires_grad = True

            # restore input. TODO, what does this do and how should it be changed when I fix the unnecessary memory storage
            xout = torch.cat([x1, x2], dim=hidden_tensor_dim)#.contiguous()
            with torch.no_grad():
                x.storage().resize_(int(np.prod(xout.shape)))
                x.set_(xout).detach()  # NOTE .detach() is very important here.

            # compute outputs building a sub-graph
            y1 = x1_stop + F_x2
            y2 = x2_stop + G_z1
            
            # calculate the final gradients for the weights and inputs
            # grad of y2 w.r.t z1 and G. y2_grad is coming from above the graph.
            dd = torch.autograd.grad(y2, (z1_stop,) + tuple(G.parameters()), y2_grad) 
            z1_grad = dd[0] + y1_grad
            GWgrads = dd[1:]

            dd = torch.autograd.grad(y1, (x1_stop, x2_stop) + tuple(F.parameters()), z1_grad)

            FWgrads = dd[2:]
            grad_input = torch.cat([dd[0], dd[1] + y2_grad], dim=hidden_tensor_dim) # positions 0 and 1 are x1 and x2 grads

            # TODO, commenting this out doesn't seem to matter                    
            y1.detach_()
            y2.detach_()
            x1.detach_()
            x2.detach_()
            z1_stop.detach_()
            G_z1.detach_()
            F_x2.detach()
            del x1, x2, y1, y2, z1_stop, G_z1, x2_stop, x1_stop

        return (grad_input, None, None, None) + FWgrads + GWgrads
