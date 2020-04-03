import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, use_bn=True,
                 actv_type='relu'):
        super(LinearLayer, self).__init__()

        """ linear layer """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            raise ValueError

    def reset_parameters(self, reset_indv_bias=None):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(0)) # kaiming init
        if (reset_indv_bias is None) or (reset_indv_bias is False):
            init.xavier_uniform_(self.weight, gain=1.0)  # xavier init
        if (reset_indv_bias is None) or ((self.bias is not None) and reset_indv_bias is True):
            init.constant_(self.bias, 0)

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        if self.bn:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)

        return out


class Conv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0,
                 bias=False, use_bn=True, 
                 pool_type=None, actv_type='relu'):
        super(Conv1DLayer, self).__init__()

        """ conv1d  layer """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_channels)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        else:
            raise ValueError

    def reset_parameters(self, reset_indv_bias=None):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(0)) # kaiming init
        if (reset_indv_bias is None) or (reset_indv_bias is False):
            init.xavier_uniform_(self.weight, gain=1.0)  # xavier init
        if (reset_indv_bias is None) or ((self.bias is not None) and reset_indv_bias is True):
            init.constant_(self.bias, 0)

    def forward(self, input):
        out = F.conv1d(input=input, weight=self.weight, bias=self.bias, 
                       stride=self.stride, padding=self.padding)
        if self.bn:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)

        return out

class vanilla_nn(nn.Module):
    def __init__(self, input_size=1, output_size=1, bias=True,
                 hidden_size=400, num_layers=4,
                 use_bn=False, actv_type='relu',
                 softmax=False):

        super(vanilla_nn, self).__init__()
        self.softmax = softmax
        self.loss = nn.MSELoss()

        self.fcs = nn.ModuleList()
        """ input layer """
        self.fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-1):
            self.fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))


    def forward(self, X):
        for layer in self.fcs:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        else:
            out = X

        return out

class prob_nn(nn.Module):
    def __init__(self, input_size=1, output_size=2, bias=True,
                 hidden_size=400, num_layers=4,
                 adversarial_eps_percent = 1,
                 use_bn=True, actv_type='relu',
                 softmax=False):

        super(prob_nn, self).__init__()
        self.softmax = softmax
        # self.loss = nn.MSELoss()
        self.mean_dim = 1

        self.fcs = nn.ModuleList()
        self.fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-1):
            self.fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))
        self.max_var = 1e6
        self.min_var = 0.0
        self.adversarial_eps_percent = adversarial_eps_percent
        self.adversarial_eps = None

    def softplus(self, x):
        softplus = torch.log(1+torch.exp(x))
        #softplus = torch.where(softplus == float('inf'), x, softplus)
        return softplus

    def determine_adversarial_eps(self, full_train_X):
        num_features = full_train_X.shape[1]
        feat_eps_tensor = torch.empty(num_features)
        for feat_idx in range(num_features):
            feat_min = torch.min(full_train_X[:,feat_idx], dim=1)
            feat_max = torch.max(full_train_X[:,feat_idx], dim=1)
            feat_range = feat_max - feat_min
            feat_eps = feat_range * self.adversarial_eps_percent / 100.
            feat_eps_tensor[feat_idx] = feat_eps
        self.adversarial_eps = feat_eps_tensor

    def loss(self, batch_pred, batch_y):
        pred_mean, pred_var = torch.split(batch_pred, self.mean_dim, dim=1)
        # pred_mean = batch_pred[:,:self.mean_dim]
        # pred_var = batch_pred[:,self.mean_dim:]

        diff = torch.sub(batch_y, pred_mean)
        for v in pred_var:
            if v == float('inf'):
                raise ValueError('infinite variance')
            if v > self.max_var:
                self.max_var = v
            if v < self.min_var:
                self.min_var = v
        loss = torch.mean(torch.div(diff**2, 2*pred_var))
        loss += torch.mean(torch.log(pred_var)/2)

        # pred_var = torch.clamp(pred_var, min=1e-10)
        # term_1 = torch.log(pred_var)/2
        # term_2 = (batch_y - pred_mean)**2/(2*pred_var)
        # loss = torch.mean(term_1 + term_2, dim=0)

        return loss

    def gen_adversarial_example(self, batch_X, grad_X):
        with torch.no_grad():
            grad_sign = torch.sign(grad_X)
            adv_ex = batch_X + (self.adversarial_eps * grad_sign)
        return adv_ex

    def nll_adversarial_loss(self, batch_X, batch_y, optimizer):
        if self.adversarial_eps is None:
            raise RuntimeError("Must run 'determine_adversarial_eps' to set eps" )

        # 1. calculate nll loss of original batch_X
        #    make it a Variable so we can get grad of loss wrt batch_X
        batch_X = Variable(batch_X)
        batch_X.requires_grad = True

        # 2. zero out gradients before calculating grad of batch_X
        optimizer.zero_grad()
        batch_pred = self.forward(batch_X)
        nll_loss = self.loss(batch_pred, batch_y)
        nll_loss.backward()
        grad_X = batch_X.grad

        # 3. make the adversarial example from batch_X
        batch_adversarial_example = self.gen_adversarial_example(batch_X, grad_X)

        # 4. no longer need to calculate gradients for batch_X and adversarial batch_X
        batch_adversarial_example.requires_grad = False
        batch_X.requires_grad = False

        # 5. calculate nll loss of adversarial batch_X
        adversarial_batch_pred = self.forward(batch_adversarial_example)
        adv_nll_loss = self.loss(adversarial_batch_pred, batch_y)

        # 6. zero out gradient before calculating final loss
        optimizer.zero_grad()

        # final loss
        batch_loss = nll_loss + adv_nll_loss

        return batch_loss



    def forward(self, X):
        for layer in self.fcs:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        else:
            out = X
        
        means = out[:,:self.mean_dim]
        variances = F.softplus(out[:,self.mean_dim:]) + 1e-8
        pnn_out = torch.cat([means, variances], dim=1)
        #pnn_out = torch.cat([out[:,:self.mean_dim], F.softplus(out[:,self.mean_dim:])], dim=1)
        return pnn_out


