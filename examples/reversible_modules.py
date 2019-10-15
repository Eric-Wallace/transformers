import torch
import numpy as np
from copy import deepcopy

hidden_tensor_dim = 2 # dimension 2 is the hidden dimension of the transformer for huggingface
clear = False

class ReversibleBlock(torch.nn.Module):
    def __init__(self, F_list, G_list):
        super(ReversibleBlock, self).__init__()
        # TODO, does this duplicate memory by storing it here?
        self.G_list = G_list
        self.F_list = F_list        

    def forward(self, inputs):
        x, attention_mask = inputs
        debug = False
        if debug: # no memory saving, used for debugging.
            out = x
            for i in range(len(self.F_list)):
                x1, x2 = torch.chunk(out, 2, dim=hidden_tensor_dim) # split the initial embeddings
                Fd = self.F_list[i].forward(x2, attention_mask)
                y1 = x1 + Fd
                Gd = self.G_list[i].forward(y1)
                y2 = x2 + Gd
                out = torch.cat([y1, y2], dim=hidden_tensor_dim)
        else:
            # TODO, by passing F_list parameters() and G_list parameters() around, does that waste memory? It seems like those parameters will also be stored inside the matrix multiplication forward() calls and such. Hopefully they all point to the same memory?
            args = [x, attention_mask, self.F_list, self.G_list]
            for F in self.F_list:
                args += [w for w in F.parameters()]
            for G in self.G_list:
                args += [w for w in G.parameters()]
            out = ReversibleBlockFunction.apply(*args)
        
        if clear:
            x.storage().resize_(0) 
        return out

class ReversibleBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, attention_mask, F_list, G_list, *weights):
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
        assert(len(F_list) == len(G_list))

        # store F and G functions in context, accessed in reverse()
        # TODO, doesn't this mean we are storing memory for the weights of F on GPU, and then we are also storing them in ctx (which is needed in reverse)? Or do they point to the same place.
        ctx.F_list = F_list
        ctx.G_list = G_list
        ctx.attention_mask = attention_mask

        with torch.no_grad():
            x1, x2 = torch.chunk(x, 2, dim=hidden_tensor_dim) # does not duplicate memory, very fast.
            for i in range(len(F_list)):
                # partition in two equally sized set of channels
                # TODO, lots of unnecessary cat and then chunking

                # compute outputs
                # TODO, can we do forwards of F and G in parallel? Will pytorch do that automatically?
                Fr = F_list[i].forward(x2, attention_mask)
                y1 = x1 + Fr
                if clear:
                    x1.set_()
                    del x1
                    Fr.set_()
                    del Fr

                Gr = G_list[i].forward(y1)
                y2 = x2 + Gr
                if clear:
                    x2.set_()
                    del x2
                    Gr.set_()
                    del Gr

                x1, x2 = y1, y2                
                # output = torch.cat([y1, y2], dim=hidden_tensor_dim)

            output = torch.cat([y1, y2], dim=hidden_tensor_dim)

        # save the input and output variables
        ctx.save_for_backward(x.data, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert(grad_output.shape[hidden_tensor_dim] % 2 == 0)

        F_list, G_list, attention_mask = ctx.F_list, ctx.G_list, ctx.attention_mask
        assert(len(F_list) == len(G_list))
        x, output = ctx.saved_tensors

        with torch.no_grad():
            y1, y2 = torch.chunk(output, 2, dim=hidden_tensor_dim)
            
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=hidden_tensor_dim)

            if clear:
                output.set_()
                del output            
                grad_output.set_()
                del grad_output

        FWgrads_list = []
        GWgrads_list = []
        for i in range(len(F_list)):                
            with torch.set_grad_enabled(True):                        
                z1_stop = y1.detach()
                z1_stop.requires_grad = True                                                
                G_z1 = G_list[i].forward(z1_stop)                
                x2 = y2 - G_z1                                

                x2_stop = x2.detach()
                x2_stop.requires_grad = True                            
                F_x2 = F_list[i].forward(x2_stop, attention_mask)
                x1 = y1 - F_x2
                x1_stop = x1.detach()
                x1_stop.requires_grad = True

                # compute outputs building a sub-graph
                y1 = x1_stop + F_x2                
                y2 = x2_stop + G_z1
                if clear:
                    F_x2.set_()
                    del F_x2
                    G_z1.set_()
                    del G_z1

                # calculate the final gradients for the weights and inputs
                # grad of y2 w.r.t z1 and G. y2_grad is coming from above the graph.            
                dd = torch.autograd.grad(y2, (z1_stop,) + tuple(G_list[i].parameters()), y2_grad)                
                GWgrads = dd[1:]                

                z1_grad = dd[0] + y1_grad                
                dd = torch.autograd.grad(y1, (x1_stop, x2_stop) + tuple(F_list[i].parameters()), z1_grad)
                x1_grad = dd[0]
                x2_grad = dd[1] + y2_grad
                FWgrads = dd[2:]                   
                                
                GWgrads_list.append(GWgrads) # TODO, is deepcopy necessary
                FWgrads_list.append(FWgrads)
                if clear:
                    del GWgrads
                    del FWgrads
            
                y1_grad = x1_grad.detach().clone() # next y1_grad, y2_grad is previous y1, y2 grad
                y2_grad = x2_grad.detach().clone()          
                if clear:
                    x1_grad.set_()
                    del x1_grad
                    x2_grad.set_()
                    del x2_grad

                    del z1_stop
                    y1.set_()
                    del y1
                    y2.set_()
                    del y2

                y1 = x1.detach().clone()                
                y2 = x2.detach().clone()            
                if clear:
                    x1.set_()
                    del x1
                    x2.set_()
                    del x2                                
                    del x2_stop                
                    del x1_stop            

        grad_input = torch.cat([y1_grad, y2_grad], dim=hidden_tensor_dim) # positions 0 and 1 are x1 and x2 grads
        if clear:
            y1_grad.set_()
            del y1_grad
            y2_grad.set_()
            del y2_grad

        # restore input
        xout = torch.cat([y1, y2], dim=hidden_tensor_dim)
        if clear:
            y1.set_()
            del y1
            y2.set_()
            del y2
        with torch.no_grad():
            x.storage().resize_(int(np.prod(xout.shape)))
            x.set_(xout).detach()  # NOTE .detach() is very important here.

        returnVal = (grad_input, None, None, None)
        for FWgrads in FWgrads_list:
            returnVal += FWgrads
        for GWgrads in GWgrads_list:
            returnVal += GWgrads
        return returnVal
