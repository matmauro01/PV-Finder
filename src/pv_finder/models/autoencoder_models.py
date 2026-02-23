#### adapted from Will's autoencoder_models file (https://github.com/Haxxardoux/pv-finder/blob/master/model/autoencoder_models.py)


import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GATConv,
    GraphConv,
    HeteroConv,
    Linear,
)


# swish custom activation
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result + sigmoid_x * (1 - result))


class Swish_module(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


class ConvBNrelu(nn.Sequential):
    """convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(p),
            #         Swish_module(),
        )


class ResConvBNrelu(nn.Module):
    """convolution => [BN] => ReLU => inplace addition of input"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__()
        assert kernel_size % 1 == 0, "even number kernel sizes will cause shape mismatch"
        self.resblock = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_channels),
            #             Swish_module(),
            nn.ReLU(),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.resblock(x) + x


class ResUp(nn.Sequential):
    """transpose convolution => convolution => [BN] => ReLU => inplace addition of input"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ResConvBNrelu(out_channels, out_channels, kernel_size=kernel_size, p=p),
        )


class Convrelu(nn.Sequential):
    """convolution => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
            ),
            nn.ReLU(),
            nn.Dropout(p),
        )


class ConvBNreluDouble(nn.Sequential):
    """convolution => [BN] => ReLU => convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            ConvBNrelu(in_channels, out_channels, kernel_size, p),
            ConvBNrelu(out_channels, out_channels, kernel_size, p),
        )


class UpnoBN(nn.Sequential):
    """transpose convolution => convolution => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            Convrelu(out_channels, out_channels, kernel_size=kernel_size, p=0),
        )


class Up(nn.Sequential):
    """transpose convolution => convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 2, 2),
            ConvBNrelu(out_channels, out_channels, kernel_size=kernel_size, p=p),
        )


class Up_alt(nn.Sequential):
    """transpose convolution => convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, in_channels, 2, 2),
            ConvBNrelu(in_channels, out_channels, kernel_size=kernel_size, p=p),
        )


class LongUp(nn.Sequential):
    """transpose convolution => ReLU convolution => [BN] => ReLU => convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, p=0):
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            ConvBNreluDouble(out_channels, out_channels, kernel_size=kernel_size, p=p),
        )


downsample_options = {
    "ConvBNrelu": ConvBNrelu,
    "ResConvBNrelu": ResConvBNrelu,
    "ConvBNreluDouble": ConvBNreluDouble,
    "Convrelu": Convrelu,
}

upsample_options = {
    "LongUp": LongUp,
    "Up": Up,
    "Up_alt": Up_alt,
    "UpnoBN": UpnoBN,
}


# ======================================================================
# DNN Models
# ======================================================================
# Used for tracks to KDE
class MaskedDNN(nn.Module):
    def __init__(
        self,
        input_size=5,
        hidden_nodes=[100, 100, 100, 100, 100],
        output_size=100,
        leaky_param=0.01,
        use_bn=False,
        use_drop=False,
        maskVal=-240.0,
        predScaleFactor=0.001,
        allow_negative_output=False,
    ):
        super().__init__()
        self.output_size = output_size
        self.LeakyReLU_param = leaky_param
        self.maskVal = maskVal
        self.predScaleFactor = predScaleFactor
        self.allow_negative_output = allow_negative_output

        # Layers
        self.linear1 = nn.Linear(input_size, hidden_nodes[0])
        self.linear2 = nn.Linear(hidden_nodes[0], hidden_nodes[1])
        self.linear3 = nn.Linear(hidden_nodes[1], hidden_nodes[2])
        self.linear4 = nn.Linear(hidden_nodes[2], hidden_nodes[3])
        self.linear5 = nn.Linear(hidden_nodes[3], hidden_nodes[4])
        self.linear6 = nn.Linear(hidden_nodes[4], output_size)

    def forward(self, x):
        leaky = nn.LeakyReLU(self.LeakyReLU_param)
        softplus = nn.Softplus()

        nEvts = x.shape[0]
        nFeatures = x.shape[1]  # noqa: F841
        nTrks = x.shape[2]  # noqa: F841

        ## --------------------------------------------------------
        ## Construct masking from the input tracks data to allow
        ## filtering only entries with tracks
        mask = x[:, 1, :] > (self.maskVal)

        ## --------------------------------------------------------
        ## Construct filter
        filt = mask.float()

        ## --------------------------------------------------------
        f1 = filt.unsqueeze(2)

        ## --------------------------------------------------------
        f2 = f1.expand(-1, -1, self.output_size)
        # print("filt.shape = ",filt.shape)
        # print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1, 2)
        # print("after transpose, x.shape = ", x.shape)

        ## --------------------------------------------------------
        ## Start forward pass here
        x = leaky(self.linear1(x))
        x = leaky(self.linear2(x))
        x = leaky(self.linear3(x))
        x = leaky(self.linear4(x))
        x = leaky(self.linear5(x))
        ## --------------------------------------------------------
        ## Output layer
        x = self.linear6(x)
        # Apply activation to output layer if not allowing negative outputs
        if not self.allow_negative_output:
            # x = leaky(x)
            # Changed to softplus
            x = softplus(x)
        # print("after softplus, x.shape = ",x.shape)

        # print("Before: x.shape = ",x.shape)
        # print("Before: f2.shape = ",f2.shape)
        ## --------------------------------------------------------
        x = x.view(nEvts, -1, self.output_size)
        # f2 = torch.unsqueeze(f2,2)
        # f2 = f2.view(nEvts,-1,self.output_size)
        ## --------------------------------------------------------
        # print("After: x1.shape = ",x.shape)
        # print("After: f2.shape = ",f2.shape)
        ## Apply masking
        x1 = torch.mul(f2, x)
        # print("After mul: x1.shape =",x1.shape)
        # print("x1.shape = ",x1.shape)

        ## --------------------------------------------------------
        x1.view(nEvts, -1, self.output_size)
        # print("After mul, view change: x1.shape =",x1.shape)
        ## --------------------------------------------------------
        ## Sum contributions from all tracks to the output (KDE here)
        y_prime = torch.sum(x1, dim=1)
        # print("Sum dim=1: x1.shape =",y_prime.shape)

        ##        print("y_prime.shape = ",y_prime.shape)

        ##        print("y_pred[:,0:10] =  ",y_pred[:,0:10])
        ##        print("y_prime[:,0:2] =  ",y_prime[:,0:10])

        ## --------------------------------------------------------
        ## Return prediction after scaling by predScaleFactor, which
        ## is meant to scale back values in a "reasonnable range",
        ## i.e. close to unity!

        y_pred = torch.mul(y_prime, self.predScaleFactor)

        return y_pred


# ======================================================================
# U-Net Models
# ======================================================================
def combine(x, y, mode="concat"):
    if mode == "concat":
        ret = torch.cat([x, y], dim=1)
        return ret

    elif mode == "add":
        return x + y
    else:
        raise RuntimeError(f"""Invalid option {mode} from choices 'concat' or 'add' """)


# ======================= UNet 100 bins ==================================== #
class UNet(nn.Module):
    def __init__(
        self,
        n=64,
        sc_mode="concat",
        dropout_p=0,
        d_selection="ConvBNrelu",
        u_selection="Up",
        n_features=4,
    ):
        super().__init__()
        if sc_mode == "concat":
            factor = 2
        else:
            factor = 1
        self.mode = sc_mode
        self.p = dropout_p

        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]

        self.rcbn1 = d_block(
            n_features, n, kernel_size=25, p=dropout_p
        )  # change to 2 if only KDEA and KDEB
        self.rcbn2 = d_block(n, n, kernel_size=7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size=5, p=dropout_p)
        # self.rcbn4 = d_block(n, n, kernel_size = 5, p=dropout_p)
        # self.rcbn5 = d_block(n, n, kernel_size = 5, p=dropout_p)

        self.up1 = u_block(n, n, kernel_size=5, p=dropout_p)
        self.up2 = u_block(n * factor, n, kernel_size=5, p=dropout_p)
        # self.up3 = u_block(n*factor, n, kernel_size = 5, p=dropout_p)
        # self.up4 = u_block(n*factor, n, kernel_size = 5, p=dropout_p)
        self.out_intermediate = nn.Conv1d(n * factor, n, 5, padding=2)
        self.outc = nn.Conv1d(n, 1, 5, padding=2)

        self.d = nn.MaxPool1d(2)

    def forward(self, x):
        # downsampling
        x1 = self.rcbn1(x)  # 100
        temp = self.rcbn2(x1)
        x2 = self.d(temp)  # 50
        temp = self.rcbn3(x2)
        x3 = self.d(temp)  # 25
        # temp = self.rcbn4(x3)
        # x4 = self.d(temp) # 1500
        # temp = self.rcbn5(x4)
        # x = self.d(temp) # 750

        x = self.up1(x3)  # 50
        temp = combine(x, x2, mode=self.mode)
        x = self.up2(temp)  # 100
        temp = combine(x, x1, mode=self.mode)
        #         x = self.up3(temp) # 6000
        #         temp = combine(x, x2, mode=self.mode)
        #         x = self.up4(temp) # 12000
        #         temp = combine(x, x1, mode=self.mode)
        x = self.out_intermediate(temp)  # 12000
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return ret

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvBNrelu):
                torch.quantization.fuse_modules(m, ["0", "1", "2"], inplace=True)


# ======================= UNetPlusPlus ==================================== #
class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        n=64,
        sc_mode="concat",
        dropout_p=0.25,
        d_selection="ConvBNrelu",
        u_selection="Up",
        n_features=4,
    ):
        super().__init__()
        if sc_mode == "concat":
            factor = 2  # noqa: F841
        else:
            factor = 1  # noqa: F841
        self.mode = sc_mode
        self.p = dropout_p

        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]  # noqa: F841

        self.rcbn1 = d_block(n_features, n, kernel_size=25, p=dropout_p)
        self.rcbn2 = d_block(n, n, kernel_size=7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size=5, p=dropout_p)
        self.rcbn4 = d_block(n, n, kernel_size=5, p=dropout_p)
        self.rcbn5 = d_block(n, n, kernel_size=5, p=dropout_p)

        self.ui = nn.ConvTranspose1d(n, n, 2, 2)
        self.i1 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.i2 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.i3 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.i4 = ConvBNrelu(3 * n, n, kernel_size=5, p=dropout_p)
        self.i5 = ConvBNrelu(3 * n, n, kernel_size=5, p=dropout_p)
        self.i6 = ConvBNrelu(4 * n, n, kernel_size=5, p=dropout_p)

        self.up1 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c1 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.up2 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c2 = ConvBNrelu(3 * n, n, kernel_size=5, p=dropout_p)
        self.up3 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c3 = ConvBNrelu(4 * n, n, kernel_size=5, p=dropout_p)
        self.up4 = nn.ConvTranspose1d(n, n, 2, 2)
        self.up_c4 = ConvBNrelu(5 * n, n, kernel_size=5, p=dropout_p)

        self.out_intermediate = nn.Conv1d(2 * n, n, 5, padding=2)  # padding=5-1//2
        self.outc = nn.Conv1d(n, 1, 5, padding=2)  # padding=5-1//2

        self.d = nn.MaxPool1d(2)

    def forward(self, x):
        # down-sampling
        d1 = self.rcbn1(x)  # 12000
        d2 = self.d(self.rcbn2(d1))  # 6000
        d3 = self.d(self.rcbn3(d2))  # 3000
        d4 = self.d(self.rcbn4(d3))  # 1500
        d5 = self.d(self.rcbn5(d4))  # 750

        # dense skip connections
        ui1 = self.ui(d2)
        ui2 = self.ui(d3)
        ui3 = self.ui(d4)

        i1 = self.i1(torch.cat([ui1, d1], dim=1))
        i2 = self.i2(torch.cat([ui2, d2], dim=1))
        i3 = self.i3(torch.cat([ui3, d3], dim=1))

        ui4 = self.ui(i2)
        ui5 = self.ui(i3)

        i4 = self.i4(torch.cat([ui4, d1, i1], dim=1))
        i5 = self.i5(torch.cat([ui5, d2, i2], dim=1))

        ui6 = self.ui(i5)

        i6 = self.i6(torch.cat([ui6, d1, i1, i4], dim=1))

        # up-sampling
        u1 = self.up_c1(torch.cat([d4, self.up1(d5)], dim=1))  # 1500
        u2 = self.up_c2(torch.cat([d3, i3, self.up2(u1)], dim=1))  # 3000
        u3 = self.up_c3(torch.cat([d2, i2, i5, self.up1(u2)], dim=1))  # 6000
        u4 = self.up_c4(torch.cat([d1, i1, i4, i6, self.up1(u3)], dim=1))  # 12000

        x = self.out_intermediate(torch.cat([u4, d1], dim=1))
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return ret

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvBNrelu):
                torch.quantization.fuse_modules(m, ["0", "1", "2"], inplace=True)


## ========================Tracks to KDE MLP + KDE to Hists UNet 100 Bins======================================================
class trackstoHists_UNet(nn.Module):
    ## Activation function to be applied to the output layer
    softplus = torch.nn.Softplus()

    def __init__(
        self,
        n_InputFeatures=5,
        n_OutputFeatures=100,
        l_HiddenNodes=[100, 100, 100, 100, 100],
        n_LatentChannels=4,
        n_UNetChannels=64,
        sc_mode="concat",
        dropout=0,
        LeakyReLU_param=0.01,
        predScaleFactor=0.001,
        maskVal=-240.0,
        d_selection="ConvBNrelu",
        u_selection="Up",
        verbose=False,
    ):
        super().__init__()

        ## *********************************************
        print("*" * 100)
        print("Initializing the trackstoHists_UNet model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =", n_InputFeatures)
        print("   - n_OutputFeatures =", n_OutputFeatures)
        print("   - l_HiddenNodes    =", l_HiddenNodes)
        print("   - n_LatentChannels =", n_LatentChannels)
        print("   - n_UNetChannels   =", n_UNetChannels)
        print("   - d_selection      =", d_selection)
        print("   - u_selection      =", u_selection)
        print("   - sc_mode          =", sc_mode)
        print("   - dropout          =", dropout)
        print("   - LeakyReLU_param  =", LeakyReLU_param)
        print("   - maskVal          =", maskVal)
        print("   - predScaleFactor  =", predScaleFactor)
        print("")
        print("*" * 100)
        ## *********************************************

        ## --------------------------------------------------------
        self.n_InputFeatures = n_InputFeatures
        self.n_OutputFeatures = n_OutputFeatures
        self.n_LatentChannels = n_LatentChannels
        self.n_UNetChannels = n_UNetChannels
        self.mode = sc_mode
        self.dropout = dropout
        self.LeakyReLU_param = LeakyReLU_param
        self.maskVal = maskVal
        self.predScaleFactor = predScaleFactor

        ## --------------------------------------------------------
        self.Nodes_L1 = l_HiddenNodes[0]
        self.Nodes_L2 = l_HiddenNodes[1]
        self.Nodes_L3 = l_HiddenNodes[2]
        self.Nodes_L4 = l_HiddenNodes[3]
        self.Nodes_L5 = l_HiddenNodes[4]

        ## --------------------------------------------------------
        self.verbose = verbose

        # ========================================================================
        # Fully Connected part of the network
        # ========================================================================

        ## --------------------------------------------------------
        self.layer1 = nn.Linear(
            in_features=n_InputFeatures, out_features=self.Nodes_L1, bias=True
        )
        self.layer2 = nn.Linear(
            in_features=self.layer1.out_features, out_features=self.Nodes_L2, bias=True
        )
        self.layer3 = nn.Linear(
            in_features=self.layer2.out_features, out_features=self.Nodes_L3, bias=True
        )
        self.layer4 = nn.Linear(
            in_features=self.layer3.out_features, out_features=self.Nodes_L4, bias=True
        )
        self.layer5 = nn.Linear(
            in_features=self.layer4.out_features, out_features=self.Nodes_L5, bias=True
        )
        ## --------------------------------------------------------
        # Output layer defined to have nBinsPerInterval output
        self.layer6A = nn.Linear(
            in_features=self.layer5.out_features,
            out_features=self.n_LatentChannels * self.n_OutputFeatures,
            bias=True,
        )

        ## ========================================================================
        ## UNet part of the network
        ## ========================================================================

        ## -----------------------------------------------------------------------
        ## General definitions
        self.relu = nn.ReLU()

        if self.mode == "concat":
            self.factor = 2
        else:
            self.factor = 1

        ## --------------------------------------------------------------------------------
        ## Make sure that if we configure the architecture using the strings, that the string is a valid choice
        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        ## --------------------------------------------------------------------------------
        ## Selection of the main component that will be use in the decoder/encoder
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Down Block 0 -> 1 (receiving input of shape [n_LatentChannels*n_OutputFeatures])
        self.rcbn1 = d_block(self.n_LatentChannels, self.n_UNetChannels, kernel_size=25)
        ## --------------------------------------------------------------------------------
        ## Down Block 1 -> 2
        self.rcbn2 = d_block(self.n_UNetChannels, self.n_UNetChannels, kernel_size=7)
        ## --------------------------------------------------------------------------------
        ## Down Block 2 -> 3
        self.rcbn3 = d_block(self.n_UNetChannels, self.n_UNetChannels, kernel_size=5)

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 3 -> 2'
        self.up1 = u_block(self.n_UNetChannels, self.n_UNetChannels, kernel_size=5)
        ## --------------------------------------------------------------------------------
        ## Up Block 2' -> 1'
        self.up2 = u_block(
            self.n_UNetChannels * self.factor, self.n_UNetChannels, kernel_size=5
        )

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 1' -> 0'
        self.out_intermediate = nn.Conv1d(
            self.n_UNetChannels * self.factor, self.n_UNetChannels, 5, padding=2
        )
        ## --------------------------------------------------------------------------------
        ## Up Block 0' -> output
        ##
        ## We need to project the n-dimensional output channels down to one,
        ## so we can call ".squeeze()" to remove it
        self.outc = nn.Conv1d(self.n_UNetChannels, 1, 5, padding=2)

        ## --------------------------------------------------------------------------------
        self.maxPool1d = nn.MaxPool1d(2)

    ## --------------------------------------------------------
    ## --------------------------------------------------------
    def forward(self, x):
        ## ====================================================
        ##  Forward pass of the Fully connected layers
        ## ====================================================

        ## --------------------------------------------------------
        ## Activation function to be applied between each hidden layer
        leakyRL = nn.LeakyReLU(self.LeakyReLU_param)

        nEvts = x.shape[0]
        nFeatures = x.shape[1]  # noqa: F841
        nTrks = x.shape[2]

        ## --------------------------------------------------------
        ## Construct masking from the input tracks data to allow
        ## filtering only entries with tracks
        # Use 1 for z_0 position
        mask = x[:, 1, :] > self.maskVal

        ## --------------------------------------------------------
        ## Construct filter
        filt = mask.float()

        ## --------------------------------------------------------
        f1 = filt.unsqueeze(2)

        ## --------------------------------------------------------
        f2 = f1.expand(-1, -1, self.n_OutputFeatures)
        # print("filt.shape = ",filt.shape)
        # print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1, 2)
        # print("after transpose, x.shape = ", x.shape)

        ## --------------------------------------------------------
        ## make a copy of the initial features so they can be passed along using a skip connection
        x0 = x  # noqa: F841
        x = leakyRL(self.layer1(x))
        x = leakyRL(self.layer2(x))
        x = leakyRL(self.layer3(x))
        x = leakyRL(self.layer4(x))
        x = leakyRL(self.layer5(x))
        ## --------------------------------------------------------
        ## produces n_LatentChannels x nBins bins for intervals
        x = F.softplus(self.layer6A(x))

        ## --------------------------------------------------------
        x = x.view(nEvts, nTrks, self.n_LatentChannels, self.n_OutputFeatures)

        ## --------------------------------------------------------
        ## here we are summing over all the tracks, creating "y"
        ## which has a sum of all tracks' contributions in each of
        ## n_LatentChannels for each event and each bin of the (eventual)
        ## KDE histogram
        ## print("before unsqueezing, f2.shape = ",f2.shape)
        f2 = torch.unsqueeze(f2, 2)
        # print("x.shape = ",x.shape)
        # print("after unsqueezing,  f2 = torch.unsqueeze(f2,2), f2,shape = ",f2.shape)
        x = torch.mul(f2, x)
        outputFCN = torch.sum(x, dim=1)
        outputFCN = torch.mul(outputFCN, self.predScaleFactor)
        # print(' after summation: outFCN.shape = ',outputFCN.shape)

        ## ====================================================
        ##  Forward pass of the UNet layers
        ## ====================================================

        ## --------------------------------------------------------
        xd1 = self.rcbn1(outputFCN)  # n_OutputFeatures
        ## --------------------------------------------------------
        xd2 = self.rcbn2(xd1)  # n_OutputFeatures
        xd2 = self.maxPool1d(xd2)  # n_OutputFeatures / 2
        ## --------------------------------------------------------
        xd3 = self.rcbn3(xd2)  # n_OutputFeatures / 2
        xd3 = self.maxPool1d(xd3)  # n_OutputFeatures / 4

        ## --------------------------------------------------------
        xu1 = self.up1(xd3)  # n_OutputFeatures / 2
        # Add a skip connection using "combine"
        xu1_skip = combine(xu1, xd2, mode=self.mode)
        ## --------------------------------------------------------
        xu2 = self.up2(xu1_skip)  # n_OutputFeatures
        # Add a skip connection using "combine"
        xu2_skip = combine(xu2, xd1, mode=self.mode)

        ## --------------------------------------------------------
        # Make an intermediate Conv layer
        x = self.out_intermediate(xu2_skip)  # n_OutputFeatures
        ## --------------------------------------------------------
        # Make final Conv layer
        logits_x = self.outc(x)

        ## --------------------------------------------------------
        # squeeze removes empty dimensions.. (n_UNetChannels, 1, n_OutputFeatures) -> (n_UNetChannels, n_OutputFeatures)
        outputs = F.softplus(logits_x).squeeze()
        # print(' after summation: outputs.shape = ',outputs.shape)
        # print('outputs sampmle', outputs[0])

        ## --------------------------------------------------------
        ## Return prediction after scaling by predScaleFactor, which
        ## is meant to scale back values in a "reasonnable range",
        ## i.e. close to unity!
        # y_pred = torch.mul(outputs,self.predScaleFactor)
        return outputs


# ======================= UNet 1000 Bins ==================================== #
class UNet_1000(nn.Module):
    def __init__(
        self,
        n=64,
        sc_mode="concat",
        dropout_p=0,
        d_selection="ConvBNrelu",
        u_selection="Up",
        n_features=4,
    ):
        super().__init__()
        if sc_mode == "concat":
            factor = 2
        else:
            factor = 1
        self.mode = sc_mode
        self.p = dropout_p

        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]

        self.rcbn1 = d_block(
            n_features, n, kernel_size=25, p=dropout_p
        )  # change to 2 if only KDEA and KDEB
        self.rcbn2 = d_block(n, n, kernel_size=7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size=5, p=dropout_p)
        self.rcbn4 = d_block(n, n, kernel_size=5, p=dropout_p)
        # self.rcbn5 = d_block(n, n, kernel_size = 5, p=dropout_p)

        self.up1 = u_block(n, n, kernel_size=5, p=dropout_p)
        self.up2 = u_block(n * factor, n, kernel_size=5, p=dropout_p)
        self.up3 = u_block(n * factor, n, kernel_size=5, p=dropout_p)
        # self.up4 = u_block(n*factor, n, kernel_size = 5, p=dropout_p)
        self.out_intermediate = nn.Conv1d(n * factor, n, 5, padding=2)
        self.outc = nn.Conv1d(n, 1, 5, padding=2)

        self.d = nn.MaxPool1d(2)

    def forward(self, x):
        # downsampling
        x1 = self.rcbn1(x)  # 1000
        x1 = self.rcbn2(x1)
        x2 = self.d(x1)  # 500
        x2 = self.rcbn3(x2)
        x3 = self.d(x2)  # 250
        x3 = self.rcbn4(x3)
        x4 = self.d(x3)  # 125

        x = self.up1(x4)  # 250
        temp = combine(x, x3, mode=self.mode)
        x = self.up2(temp)  # 500
        temp = combine(x, x2, mode=self.mode)
        x = self.up3(temp)  # 1000
        temp = combine(x, x1, mode=self.mode)
        x = self.out_intermediate(temp)  # 1000
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return ret


# ======================= MLP+UNet 1000 Bins ==================================== #
class trackstoHists_UNet_1000(nn.Module):
    ## Activation function to be applied to the output layer
    softplus = torch.nn.Softplus()

    def __init__(
        self,
        n_InputFeatures=7,
        n_OutputFeatures=1000,
        l_HiddenNodes=[100, 100, 100, 100, 100],
        n_LatentChannels=8,
        n_UNetChannels=64,
        sc_mode="concat",
        dropout=0.25,
        LeakyReLU_param=0.01,
        predScaleFactor=0.001,
        maskVal=-240.0,
        d_selection="ConvBNrelu",
        u_selection="Up",
        verbose=False,
    ):
        super().__init__()

        ## *********************************************
        print("*" * 100)
        print("Initializing the trackstoHists_UNet model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =", n_InputFeatures)
        print("   - n_OutputFeatures =", n_OutputFeatures)
        print("   - l_HiddenNodes    =", l_HiddenNodes)
        print("   - n_LatentChannels =", n_LatentChannels)
        print("   - n_UNetChannels   =", n_UNetChannels)
        print("   - d_selection      =", d_selection)
        print("   - u_selection      =", u_selection)
        print("   - sc_mode          =", sc_mode)
        print("   - dropout          =", dropout)
        print("   - LeakyReLU_param  =", LeakyReLU_param)
        print("   - maskVal          =", maskVal)
        print("   - predScaleFactor  =", predScaleFactor)
        print("")
        print("*" * 100)
        ## *********************************************

        ## --------------------------------------------------------
        self.n_InputFeatures = n_InputFeatures
        self.n_OutputFeatures = n_OutputFeatures
        self.n_LatentChannels = n_LatentChannels
        self.n_UNetChannels = n_UNetChannels
        self.mode = sc_mode
        self.dropout = dropout
        self.LeakyReLU_param = LeakyReLU_param
        self.maskVal = maskVal
        self.predScaleFactor = predScaleFactor

        ## --------------------------------------------------------
        self.Nodes_L1 = l_HiddenNodes[0]
        self.Nodes_L2 = l_HiddenNodes[1]
        self.Nodes_L3 = l_HiddenNodes[2]
        self.Nodes_L4 = l_HiddenNodes[3]
        self.Nodes_L5 = l_HiddenNodes[4]

        ## --------------------------------------------------------
        self.verbose = verbose

        # ========================================================================
        # Fully Connected part of the network
        # ========================================================================

        ## --------------------------------------------------------
        self.layer1 = nn.Linear(
            in_features=n_InputFeatures, out_features=self.Nodes_L1, bias=True
        )
        self.layer2 = nn.Linear(
            in_features=self.layer1.out_features, out_features=self.Nodes_L2, bias=True
        )
        self.layer3 = nn.Linear(
            in_features=self.layer2.out_features, out_features=self.Nodes_L3, bias=True
        )
        self.layer4 = nn.Linear(
            in_features=self.layer3.out_features, out_features=self.Nodes_L4, bias=True
        )
        self.layer5 = nn.Linear(
            in_features=self.layer4.out_features, out_features=self.Nodes_L5, bias=True
        )
        ## --------------------------------------------------------
        # Output layer defined to have nBinsPerInterval output
        self.layer6A = nn.Linear(
            in_features=self.layer5.out_features,
            out_features=self.n_LatentChannels * self.n_OutputFeatures,
            bias=True,
        )

        ## ========================================================================
        ## UNet part of the network
        ## ========================================================================

        ## -----------------------------------------------------------------------
        ## General definitions
        self.relu = nn.ReLU()

        if self.mode == "concat":
            self.factor = 2
        else:
            self.factor = 1

        ## --------------------------------------------------------------------------------
        ## Make sure that if we configure the architecture using the strings, that the string is a valid choice
        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        ## --------------------------------------------------------------------------------
        ## Selection of the main component that will be use in the decoder/encoder
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Down Block 0 -> 1 (receiving input of shape [n_LatentChannels*n_OutputFeatures])
        self.rcbn1 = d_block(
            self.n_LatentChannels, self.n_UNetChannels, kernel_size=25, p=self.dropout
        )
        ## --------------------------------------------------------------------------------
        ## Down Block 1 -> 2
        self.rcbn2 = d_block(
            self.n_UNetChannels, self.n_UNetChannels, kernel_size=7, p=self.dropout
        )
        ## --------------------------------------------------------------------------------
        ## Down Block 2 -> 3
        self.rcbn3 = d_block(
            self.n_UNetChannels, self.n_UNetChannels, kernel_size=5, p=self.dropout
        )
        ## --------------------------------------------------------------------------------
        ## Down Block 3 -> 4
        self.rcbn4 = d_block(
            self.n_UNetChannels, self.n_UNetChannels, kernel_size=5, p=self.dropout
        )

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 4 -> 3'
        self.up1 = u_block(
            self.n_UNetChannels, self.n_UNetChannels, kernel_size=5, p=self.dropout
        )
        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 3' -> 2'
        self.up2 = u_block(
            self.n_UNetChannels * self.factor,
            self.n_UNetChannels,
            kernel_size=5,
            p=self.dropout,
        )
        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 2' -> 1'
        self.up3 = u_block(
            self.n_UNetChannels * self.factor,
            self.n_UNetChannels,
            kernel_size=5,
            p=self.dropout,
        )

        ## --------------------------------------------------------------------------------
        ## --------------------------------------------------------------------------------
        ## Up Block 1' -> 0'
        self.out_intermediate = nn.Conv1d(
            self.n_UNetChannels * self.factor, self.n_UNetChannels, 5, padding=2
        )
        ## --------------------------------------------------------------------------------
        ## Up Block 0' -> output
        ##
        ## We need to project the n-dimensional output channels down to one,
        ## so we can call ".squeeze()" to remove it
        self.outc = nn.Conv1d(self.n_UNetChannels, 1, 5, padding=2)

        ## --------------------------------------------------------------------------------
        self.maxPool1d = nn.MaxPool1d(2)

    ## --------------------------------------------------------
    ## --------------------------------------------------------
    def forward(self, x):
        ## ====================================================
        ##  Forward pass of the Fully connected layers
        ## ====================================================

        ## --------------------------------------------------------
        ## Activation function to be applied between each hidden layer
        leakyRL = nn.LeakyReLU(self.LeakyReLU_param)

        nEvts = x.shape[0]
        nFeatures = x.shape[1]  # noqa: F841
        nTrks = x.shape[2]

        ## --------------------------------------------------------
        ## Construct masking from the input tracks data to allow
        ## filtering only entries with tracks
        # Use 1 for z_0 position
        mask = x[:, 1, :] > self.maskVal

        ## --------------------------------------------------------
        ## Construct filter
        filt = mask.float()

        ## --------------------------------------------------------
        f1 = filt.unsqueeze(2)

        ## --------------------------------------------------------
        f2 = f1.expand(-1, -1, self.n_OutputFeatures)
        # print("filt.shape = ",filt.shape)
        # print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1, 2)
        # print("after transpose, x.shape = ", x.shape)

        ## --------------------------------------------------------
        ## make a copy of the initial features so they can be passed along using a skip connection
        x0 = x  # noqa: F841
        x = leakyRL(self.layer1(x))
        x = leakyRL(self.layer2(x))
        x = leakyRL(self.layer3(x))
        x = leakyRL(self.layer4(x))
        x = leakyRL(self.layer5(x))
        ## --------------------------------------------------------
        ## produces n_LatentChannels x nBins bins for intervals
        x = F.softplus(self.layer6A(x))

        ## --------------------------------------------------------
        x = x.view(nEvts, nTrks, self.n_LatentChannels, self.n_OutputFeatures)

        ## --------------------------------------------------------
        ## here we are summing over all the tracks, creating "y"
        ## which has a sum of all tracks' contributions in each of
        ## n_LatentChannels for each event and each bin of the (eventual)
        ## KDE histogram
        ## print("before unsqueezing, f2.shape = ",f2.shape)
        f2 = torch.unsqueeze(f2, 2)
        # print("x.shape = ",x.shape)
        # print("after unsqueezing,  f2 = torch.unsqueeze(f2,2), f2,shape = ",f2.shape)
        x = torch.mul(f2, x)
        outputFCN = torch.sum(x, dim=1)
        outputFCN = torch.mul(outputFCN, self.predScaleFactor)
        # print(' after summation: outFCN.shape = ',outputFCN.shape)

        ## ====================================================
        ##  Forward pass of the UNet layers
        ## ====================================================

        ## --------------------------------------------------------
        xd1 = self.rcbn1(outputFCN)  # n_OutputFeatures
        ## --------------------------------------------------------
        xd1 = self.rcbn2(xd1)  # n_OutputFeatures
        xd2 = self.maxPool1d(xd1)  # n_OutputFeatures / 2
        ## --------------------------------------------------------
        xd2 = self.rcbn3(xd2)  # n_OutputFeatures / 2
        xd3 = self.maxPool1d(xd2)  # n_OutputFeatures / 4
        ## --------------------------------------------------------
        xd3 = self.rcbn4(xd3)  # n_OutputFeatures / 4
        xd4 = self.maxPool1d(xd3)  # n_OutputFeatures / 8

        ## --------------------------------------------------------
        xu1 = self.up1(xd4)  # n_OutputFeatures / 4
        # Add a skip connection using "combine"
        xu1_skip = combine(xu1, xd3, mode=self.mode)
        ## --------------------------------------------------------
        xu2 = self.up2(xu1_skip)  # n_OutputFeatures / 2
        # Add a skip connection using "combine"
        xu2_skip = combine(xu2, xd2, mode=self.mode)

        ## --------------------------------------------------------
        xu3 = self.up3(xu2_skip)  # n_OutputFeatures / 2
        # Add a skip connection using "combine"
        xu3_skip = combine(xu3, xd1, mode=self.mode)

        ## --------------------------------------------------------
        # Make an intermediate Conv layer
        x = self.out_intermediate(xu3_skip)  # n_OutputFeatures
        ## --------------------------------------------------------
        # Make final Conv layer
        logits_x = self.outc(x)

        ## --------------------------------------------------------
        # squeeze removes empty dimensions.. (n_UNetChannels, 1, n_OutputFeatures) -> (n_UNetChannels, n_OutputFeatures)
        outputs = F.softplus(logits_x).squeeze()
        # print(' after summation: outputs.shape = ',outputs.shape)
        # print('outputs sampmle', outputs[0])

        ## --------------------------------------------------------
        ## Return prediction after scaling by predScaleFactor, which
        ## is meant to scale back values in a "reasonnable range",
        ## i.e. close to unity!
        # y_pred = torch.mul(outputs,self.predScaleFactor)
        return outputs


# ======================= UNetPlusPlus 1000 Bins ==================================== #
class UNetPlusPlus_1000(nn.Module):
    def __init__(
        self,
        n=64,
        sc_mode="concat",
        dropout_p=0.25,
        d_selection="ConvBNrelu",
        u_selection="Up",
        n_features=1,
    ):
        super().__init__()
        if sc_mode == "concat":
            factor = 2  # noqa: F841
        else:
            factor = 1  # noqa: F841
        self.mode = sc_mode
        self.p = dropout_p

        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]  # noqa: F841

        self.rcbn1 = d_block(n_features, n, kernel_size=25, p=dropout_p)
        self.rcbn2 = d_block(n, n, kernel_size=7, p=dropout_p)
        self.rcbn3 = d_block(n, n, kernel_size=5, p=dropout_p)
        self.rcbn4 = d_block(n, n, kernel_size=5, p=dropout_p)
        self.rcbn5 = d_block(n, n, kernel_size=5, p=dropout_p)

        self.ui = nn.ConvTranspose1d(n, n, 2, 2)
        self.i1 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.i2 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.i3 = ConvBNrelu(2 * n, n, kernel_size=5, p=dropout_p)
        self.i4 = ConvBNrelu(3 * n, n, kernel_size=5, p=dropout_p)
        self.i5 = ConvBNrelu(3 * n, n, kernel_size=5, p=dropout_p)
        self.i6 = ConvBNrelu(4 * n, n, kernel_size=5, p=dropout_p)

        self.out_intermediate = nn.Conv1d(n, n, 5, padding=2)  # padding=5-1//2
        self.outc = nn.Conv1d(n, 1, 5, padding=2)  # padding=5-1//2

        self.d = nn.MaxPool1d(2)

    def forward(self, x):
        # down-sampling
        d1 = self.rcbn1(x)  # 1000 (x0,0)
        d2 = self.d(self.rcbn2(d1))  # 500
        d3 = self.d(self.rcbn3(d2))  # 250
        d4 = self.d(self.rcbn4(d3))  # 125

        # up-sampling
        ui1 = self.ui(d2)
        ui2 = self.ui(d3)
        ui3 = self.ui(d4)

        # skip-connections + upsampling
        i1 = self.i1(torch.cat([ui1, d1], dim=1))  # x(0,1)
        i2 = self.i2(torch.cat([ui2, d2], dim=1))  # x(1,1)
        i3 = self.i3(torch.cat([ui3, d3], dim=1))  # x(2,1)

        # up-sampling
        ui4 = self.ui(i2)
        ui5 = self.ui(i3)

        # skip-connections + upsampling
        i4 = self.i4(torch.cat([ui4, d1, i1], dim=1))  # x(0,2)
        i5 = self.i5(torch.cat([ui5, d2, i2], dim=1))  # x(1,2)

        # upsampling
        ui6 = self.ui(i5)

        # skip-connections + upsampling
        i6 = self.i6(torch.cat([ui6, d1, i1, i4], dim=1))  # x(0,3)

        x = self.out_intermediate(i6)
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return ret


# ======================= MLP+UNetPlusPlus 1000 Bins ==================================== #
class trackstoHists_UNetPlusPlus_1000(nn.Module):
    ## Activation function to be applied to the output layer
    softplus = torch.nn.Softplus()

    def __init__(
        self,
        n_InputFeatures=7,
        n_OutputFeatures=1000,
        l_HiddenNodes=[100, 100, 100, 100, 100],
        n_LatentChannels=8,
        n_UNetChannels=64,
        sc_mode="concat",
        dropout=0.25,
        LeakyReLU_param=0.01,
        predScaleFactor=0.001,
        maskVal=-240.0,
        d_selection="ConvBNrelu",
        u_selection="Up",
        verbose=False,
    ):
        super().__init__()

        ## *********************************************
        print("*" * 100)
        print("Initializing the trackstoHists_UNet model\n")
        print("")
        print("with the following parameters:\n")
        print("   - n_InputFeatures  =", n_InputFeatures)
        print("   - n_OutputFeatures =", n_OutputFeatures)
        print("   - l_HiddenNodes    =", l_HiddenNodes)
        print("   - n_LatentChannels =", n_LatentChannels)
        print("   - n_UNetChannels   =", n_UNetChannels)
        print("   - d_selection      =", d_selection)
        print("   - u_selection      =", u_selection)
        print("   - sc_mode          =", sc_mode)
        print("   - dropout          =", dropout)
        print("   - LeakyReLU_param  =", LeakyReLU_param)
        print("   - maskVal          =", maskVal)
        print("   - predScaleFactor  =", predScaleFactor)
        print("")
        print("*" * 100)
        ## *********************************************

        ## --------------------------------------------------------
        self.n_InputFeatures = n_InputFeatures
        self.n_OutputFeatures = n_OutputFeatures
        self.n_LatentChannels = n_LatentChannels
        self.n_UNetChannels = n_UNetChannels
        self.mode = sc_mode
        self.dropout = dropout
        self.LeakyReLU_param = LeakyReLU_param
        self.maskVal = maskVal
        self.predScaleFactor = predScaleFactor

        ## --------------------------------------------------------
        self.Nodes_L1 = l_HiddenNodes[0]
        self.Nodes_L2 = l_HiddenNodes[1]
        self.Nodes_L3 = l_HiddenNodes[2]
        self.Nodes_L4 = l_HiddenNodes[3]
        self.Nodes_L5 = l_HiddenNodes[4]

        ## --------------------------------------------------------
        self.verbose = verbose

        # ========================================================================
        # Fully Connected part of the network
        # ========================================================================

        ## --------------------------------------------------------
        self.layer1 = nn.Linear(
            in_features=n_InputFeatures, out_features=self.Nodes_L1, bias=True
        )
        self.layer2 = nn.Linear(
            in_features=self.layer1.out_features, out_features=self.Nodes_L2, bias=True
        )
        self.layer3 = nn.Linear(
            in_features=self.layer2.out_features, out_features=self.Nodes_L3, bias=True
        )
        self.layer4 = nn.Linear(
            in_features=self.layer3.out_features, out_features=self.Nodes_L4, bias=True
        )
        self.layer5 = nn.Linear(
            in_features=self.layer4.out_features, out_features=self.Nodes_L5, bias=True
        )
        ## --------------------------------------------------------
        # Output layer defined to have nBinsPerInterval output
        self.layer6A = nn.Linear(
            in_features=self.layer5.out_features,
            out_features=self.n_LatentChannels * self.n_OutputFeatures,
            bias=True,
        )

        ## ========================================================================
        ## UNet part of the network
        ## ========================================================================

        ## -----------------------------------------------------------------------
        ## General definitions
        self.relu = nn.ReLU()

        if self.mode == "concat":
            self.factor = 2
        else:
            self.factor = 1

        ## --------------------------------------------------------------------------------
        ## Make sure that if we configure the architecture using the strings, that the string is a valid choice
        assert d_selection in downsample_options.keys(), (
            f"Selection for downsampling block {d_selection} not present in available options - {downsample_options.keys()}"
        )
        assert u_selection in upsample_options.keys(), (
            f"Selection for downsampling block {u_selection} not present in available options - {upsample_options.keys()}"
        )

        ## --------------------------------------------------------------------------------
        ## Selection of the main component that will be use in the decoder/encoder
        d_block = downsample_options[d_selection]
        u_block = upsample_options[u_selection]  # noqa: F841

        self.rcbn1 = d_block(
            n_LatentChannels, n_UNetChannels, kernel_size=25, p=dropout
        )
        self.rcbn2 = d_block(n_UNetChannels, n_UNetChannels, kernel_size=7, p=dropout)
        self.rcbn3 = d_block(n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout)
        self.rcbn4 = d_block(n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout)
        self.rcbn5 = d_block(n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout)

        self.ui = nn.ConvTranspose1d(n_UNetChannels, n_UNetChannels, 2, 2)
        self.i1 = ConvBNrelu(
            2 * n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout
        )
        self.i2 = ConvBNrelu(
            2 * n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout
        )
        self.i3 = ConvBNrelu(
            2 * n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout
        )
        self.i4 = ConvBNrelu(
            3 * n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout
        )
        self.i5 = ConvBNrelu(
            3 * n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout
        )
        self.i6 = ConvBNrelu(
            4 * n_UNetChannels, n_UNetChannels, kernel_size=5, p=dropout
        )

        self.out_intermediate = nn.Conv1d(
            n_UNetChannels, n_UNetChannels, 5, padding=2
        )  # padding=5-1//2
        self.outc = nn.Conv1d(n_UNetChannels, 1, 5, padding=2)  # padding=5-1//2

        self.d = nn.MaxPool1d(2)

    ## --------------------------------------------------------
    ## --------------------------------------------------------
    def forward(self, x):
        ## ====================================================
        ##  Forward pass of the Fully connected layers
        ## ====================================================

        ## --------------------------------------------------------
        ## Activation function to be applied between each hidden layer
        leakyRL = nn.LeakyReLU(self.LeakyReLU_param)

        nEvts = x.shape[0]
        nFeatures = x.shape[1]  # noqa: F841
        nTrks = x.shape[2]

        ## --------------------------------------------------------
        ## Construct masking from the input tracks data to allow
        ## filtering only entries with tracks
        # Use 1 for z_0 position
        mask = x[:, 1, :] > self.maskVal

        ## --------------------------------------------------------
        ## Construct filter
        filt = mask.float()

        ## --------------------------------------------------------
        f1 = filt.unsqueeze(2)

        ## --------------------------------------------------------
        f2 = f1.expand(-1, -1, self.n_OutputFeatures)
        # print("filt.shape = ",filt.shape)
        # print("f1.shape = ",f1.shape, "f2.shape = ",f2.shape)
        x = x.transpose(1, 2)
        # print("after transpose, x.shape = ", x.shape)

        ## --------------------------------------------------------
        ## make a copy of the initial features so they can be passed along using a skip connection
        x0 = x  # noqa: F841
        x = leakyRL(self.layer1(x))
        x = leakyRL(self.layer2(x))
        x = leakyRL(self.layer3(x))
        x = leakyRL(self.layer4(x))
        x = leakyRL(self.layer5(x))
        ## --------------------------------------------------------
        ## produces n_LatentChannels x nBins bins for intervals
        x = F.softplus(self.layer6A(x))

        ## --------------------------------------------------------
        x = x.view(nEvts, nTrks, self.n_LatentChannels, self.n_OutputFeatures)

        ## --------------------------------------------------------
        ## here we are summing over all the tracks, creating "y"
        ## which has a sum of all tracks' contributions in each of
        ## n_LatentChannels for each event and each bin of the (eventual)
        ## KDE histogram
        ## print("before unsqueezing, f2.shape = ",f2.shape)
        f2 = torch.unsqueeze(f2, 2)
        # print("x.shape = ",x.shape)
        # print("after unsqueezing,  f2 = torch.unsqueeze(f2,2), f2,shape = ",f2.shape)
        x = torch.mul(f2, x)
        outputFCN = torch.sum(x, dim=1)
        outputFCN = torch.mul(outputFCN, self.predScaleFactor)
        # print(' after summation: outFCN.shape = ',outputFCN.shape)

        # down-sampling
        d1 = self.rcbn1(outputFCN)  # 1000 (x0,0)
        d2 = self.d(self.rcbn2(d1))  # 500
        d3 = self.d(self.rcbn3(d2))  # 250
        d4 = self.d(self.rcbn4(d3))  # 125

        # up-sampling
        ui1 = self.ui(d2)
        ui2 = self.ui(d3)
        ui3 = self.ui(d4)

        # skip-connections + upsampling
        i1 = self.i1(torch.cat([ui1, d1], dim=1))  # x(0,1)
        i2 = self.i2(torch.cat([ui2, d2], dim=1))  # x(1,1)
        i3 = self.i3(torch.cat([ui3, d3], dim=1))  # x(2,1)

        # up-sampling
        ui4 = self.ui(i2)
        ui5 = self.ui(i3)

        # skip-connections + upsampling
        i4 = self.i4(torch.cat([ui4, d1, i1], dim=1))  # x(0,2)
        i5 = self.i5(torch.cat([ui5, d2, i2], dim=1))  # x(1,2)

        # upsampling
        ui6 = self.ui(i5)

        # skip-connections + upsampling
        i6 = self.i6(torch.cat([ui6, d1, i1, i4], dim=1))  # x(0,3)

        x = self.out_intermediate(i6)
        logits_x0 = self.outc(x)

        ret = F.softplus(logits_x0).squeeze()
        return ret


# ======================= Perturbative UNet ==================================== #
class PerturbativeUNet(nn.Module):
    def __init__(self, args, n, sc_mode="concat", dropout_p=0):
        super().__init__()
        self.mode = sc_mode
        if sc_mode == "concat":
            factor = 2
        else:
            factor = 1

        # network for perturbative features
        self.cbn1_x = ConvBNrelu(2, n, kernel_size=11, p=dropout_p)
        self.cbn2_x = ConvBNrelu(n, n, p=dropout_p)
        self.cbn3_x = ConvBNrelu(n, n, p=dropout_p)
        self.cbn4_x = ConvBNrelu(n, n, p=dropout_p)
        self.up1_x = Up(n, n, p=dropout_p)
        self.up2_x = Up(n * factor, n, p=dropout_p)
        self.up3_x = Up(n * factor, n, p=dropout_p)
        self.up4_x = Up(n * factor, n, p=dropout_p)

        self.down = nn.MaxPool1d(2)
        self.d = nn.MaxPool1d(2)

        # network for X features
        self.rcbn1 = ConvBNrelu(1, n, kernel_size=25, p=dropout_p)
        self.rcbn2 = ConvBNrelu(n, n, kernel_size=7, p=dropout_p)
        self.rcbn3 = ConvBNrelu(n, n, kernel_size=5, p=dropout_p)
        self.rcbn4 = ConvBNrelu(n, n, kernel_size=5, p=dropout_p)
        self.rcbn5 = ConvBNrelu(n, n, kernel_size=5, p=dropout_p)

        self.up1 = Up(n, n, kernel_size=5, p=dropout_p)
        self.up2 = Up(n * factor, n, kernel_size=5, p=dropout_p)
        self.up3 = Up(n * factor, n, kernel_size=5, p=dropout_p)
        self.up4 = Up(n * factor, n, kernel_size=5, p=dropout_p)
        self.out_intermediate = nn.Conv1d(n * factor, n, 5, padding=2)

        self.outc_larger = nn.Conv1d(factor * n, 1, 3, padding=1)

    def forward(self, x):
        X = x[:, 0:1, :]  # one-slice prevents need for .unsqueeze()
        x_y = x[:, -2:, :]

        # x / y  feature
        p_x = self.cbn1_x(x_y)
        p_x = self.down(p_x)

        p_x2 = self.cbn2_x(p_x)
        p_x = self.down(p_x2)

        p_x3 = self.cbn3_x(p_x)
        p_x = self.down(p_x3)

        p_x4 = self.cbn4_x(p_x)
        p_x = self.down(p_x4)

        p_x = self.up1_x(p_x)
        p_x = self.up2_x(combine(p_x, p_x4, mode=self.mode))
        p_x = self.up3_x(combine(p_x, p_x3, mode=self.mode))
        logits_x1 = self.up4_x(combine(p_x, p_x2, mode=self.mode))

        # X feature based on U-Net (parallel network)
        x1 = self.rcbn1(X)  # 4000
        x2 = self.d(self.rcbn2(x1))  # 2000
        x3 = self.d(self.rcbn3(x2))  # 1000
        x4 = self.d(self.rcbn4(x3))  # 500
        x = self.d(self.rcbn5(x4))  # 250

        x = self.up1(x)  # 500
        x = self.up2(combine(x, x4, mode=self.mode))  # 1000
        x = self.up3(combine(x, x3, mode=self.mode))  # 2000
        x = self.up4(combine(x, x2, mode=self.mode))  # 4000
        logits_x0 = self.out_intermediate(combine(x, x1, mode=self.mode))

        logits_X_and_x = self.outc_larger(combine(logits_x0, logits_x1, mode=self.mode))

        ret = F.softplus(logits_X_and_x).squeeze(1)
        return ret


# ======================= TTVA GraphConv Model ==================================== #
class TTVA_GraphConv_Model(torch.nn.Module):
    def __init__(
        self,
        track_input_size=7,
        pv_input_size=1,
        hidden_dim=32,
        leaky_param=0.01,
        dropout=0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.LeakyReLU_param = leaky_param
        self.track_encoder = Linear(track_input_size, hidden_dim)
        self.pv_encoder = Linear(pv_input_size, hidden_dim)

        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("track", "to", "pv"): GraphConv((-1, -1), hidden_dim),
                        ("pv", "rev_to", "track"): GraphConv((-1, -1), hidden_dim),
                    },
                    aggr="max",
                )
            )

        # Final edge prediction layers
        self.intermediary_layer_1 = Linear(2 * hidden_dim, hidden_dim)
        self.intermediary_layer_2 = Linear(hidden_dim, hidden_dim)
        self.edge_predictor = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        leaky = nn.LeakyReLU(self.LeakyReLU_param)
        # Encode initial features
        track_init = leaky(self.track_encoder(data["track"].x.float()))
        pv_init = leaky(self.pv_encoder(data["pv"].x.float()))

        x_dict = {"track": track_init, "pv": pv_init}

        # Propagate through GNN layers with identity mapping
        for conv in self.convs:
            x_dict_new = conv(x_dict, data.edge_index_dict)
            x_dict = {
                node_type: F.relu(x_dict_new[node_type] + x_dict[node_type])
                for node_type in x_dict
            }

        # Edge prediction
        src, dst = data[("track", "to", "pv")].edge_index
        edge_feat = torch.cat([x_dict["track"][src], x_dict["pv"][dst]], dim=1)
        intermed_1 = self.intermediary_layer_1(edge_feat)
        intermed_1 = leaky(intermed_1)

        intermed_2 = self.intermediary_layer_2(intermed_1)
        intermed_2 = leaky(intermed_2)

        edge_logits = self.edge_predictor(intermed_2).squeeze()
        return edge_logits


# ======================= TTVA GATConv Model + EdgeAttr ==================================== #
class TTVA_GATGraphConv_Model(torch.nn.Module):
    def __init__(
        self,
        track_input_size=7,
        pv_input_size=2,
        hidden_dim=32,
        leaky_param=0.01,
        dropout=0.25,
        num_heads=4,
        edge_attr_dim=1,
    ):
        super().__init__()
        self.dropout = dropout
        self.LeakyReLU_param = leaky_param
        self.track_encoder = Linear(track_input_size, hidden_dim)
        self.pv_encoder = Linear(pv_input_size, hidden_dim)
        self.num_heads = num_heads
        self.edge_attr_dim = edge_attr_dim

        self.num_layers = 2
        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        ("track", "to", "pv"): GATConv(
                            in_channels=(-1, -1),
                            out_channels=hidden_dim,
                            heads=self.num_heads,
                            concat=False,
                            edge_dim=self.edge_attr_dim,
                            add_self_loops=False,
                        ),
                        ("pv", "rev_to", "track"): GATConv(
                            in_channels=(-1, -1),
                            out_channels=hidden_dim,
                            heads=self.num_heads,
                            concat=False,
                            edge_dim=self.edge_attr_dim,
                            add_self_loops=False,
                        ),
                    },
                    aggr="max",
                )
            )

        # Final edge prediction layers
        self.intermediary_layer_1 = Linear(2 * hidden_dim, hidden_dim)
        self.intermediary_layer_2 = Linear(hidden_dim, hidden_dim)
        self.edge_predictor = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        leaky = nn.LeakyReLU(self.LeakyReLU_param)
        # Encode initial features
        track_init = leaky(self.track_encoder(data["track"].x.float()))
        pv_init = leaky(self.pv_encoder(data["pv"].x.float()))

        x_dict = {"track": track_init, "pv": pv_init}

        # Propagate through GNN layers with identity mapping
        for conv in self.convs:
            x_dict_new = conv(
                x_dict, data.edge_index_dict, edge_attr_dict=data.edge_attr_dict
            )
            x_dict = {
                node_type: F.relu(x_dict_new[node_type] + x_dict[node_type])
                for node_type in x_dict
            }

        # Edge prediction
        src, dst = data[("track", "to", "pv")].edge_index
        edge_feat = torch.cat([x_dict["track"][src], x_dict["pv"][dst]], dim=1)
        intermed_1 = self.intermediary_layer_1(edge_feat)
        intermed_1 = leaky(intermed_1)

        intermed_2 = self.intermediary_layer_2(intermed_1)
        intermed_2 = leaky(intermed_2)

        edge_logits = self.edge_predictor(intermed_2).squeeze()
        return edge_logits


# ======================= MLP-Only Model (No UNet) ==================================== #
class TracksToHist_MLP_1000(nn.Module):
    """
    Standalone MLP model that predicts histograms directly from tracks.
    No UNet - just the MLP layers from trackstoHists_UNet_1000.

    This class is designed to be saved and loaded for evaluation, where
    calling model(inputs) returns the histogram prediction directly.
    """

    def __init__(
        self,
        n_InputFeatures: int = 7,
        n_OutputFeatures: int = 1000,
        l_HiddenNodes: list = None,
        n_LatentChannels: int = 1,
        dropout: float = 0.25,
        LeakyReLU_param: float = 0.01,
        predScaleFactor: float = 0.001,
        maskVal: float = -240.0,
    ):
        super().__init__()

        if l_HiddenNodes is None:
            l_HiddenNodes = [100, 100, 100, 100, 100]

        self.n_InputFeatures = n_InputFeatures
        self.n_OutputFeatures = n_OutputFeatures
        self.n_LatentChannels = n_LatentChannels
        self.LeakyReLU_param = LeakyReLU_param
        self.predScaleFactor = predScaleFactor
        self.maskVal = maskVal

        # MLP layers (same as in trackstoHists_UNet_1000)
        self.layer1 = nn.Linear(n_InputFeatures, l_HiddenNodes[0])
        self.layer2 = nn.Linear(l_HiddenNodes[0], l_HiddenNodes[1])
        self.layer3 = nn.Linear(l_HiddenNodes[1], l_HiddenNodes[2])
        self.layer4 = nn.Linear(l_HiddenNodes[2], l_HiddenNodes[3])
        self.layer5 = nn.Linear(l_HiddenNodes[3], l_HiddenNodes[4])
        self.layer6A = nn.Linear(l_HiddenNodes[4], n_LatentChannels * n_OutputFeatures)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        print("=" * 80)
        print("Initialized TracksToHist_MLP_1000 (MLP-only, no UNet)")
        print(f"  n_InputFeatures:  {n_InputFeatures}")
        print(f"  n_OutputFeatures: {n_OutputFeatures}")
        print(f"  n_LatentChannels: {n_LatentChannels}")
        print(f"  l_HiddenNodes:    {l_HiddenNodes}")
        print(f"  dropout:          {dropout}")
        print(f"  predScaleFactor:  {predScaleFactor}")
        print("=" * 80)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP only.

        Args:
            x: Input tensor of shape (batch, n_features, n_tracks)

        Returns:
            Histogram prediction of shape (batch, n_OutputFeatures)
        """
        leaky_relu = nn.LeakyReLU(self.LeakyReLU_param)

        n_events, _, n_tracks = x.shape

        # Masking based on z_0 position
        mask = x[:, 1, :] > self.maskVal
        filt = mask.float()
        f1 = filt.unsqueeze(2)
        f2 = f1.expand(-1, -1, self.n_OutputFeatures)

        # Transpose for linear layers: (batch, n_tracks, n_features)
        x = x.transpose(1, 2)

        # MLP forward pass
        x = leaky_relu(self.layer1(x))
        x = leaky_relu(self.layer2(x))
        x = leaky_relu(self.layer3(x))
        x = leaky_relu(self.layer4(x))
        x = leaky_relu(self.layer5(x))
        x = F.softplus(self.layer6A(x))

        # Reshape to (batch, n_tracks, n_LatentChannels, n_OutputFeatures)
        x = x.view(n_events, n_tracks, self.n_LatentChannels, self.n_OutputFeatures)

        # Apply mask and sum over tracks
        f2 = torch.unsqueeze(f2, 2)
        x = torch.mul(f2, x)
        output = torch.sum(x, dim=1)  # (batch, n_LatentChannels, n_OutputFeatures)
        output = torch.mul(output, float(self.predScaleFactor))

        # Squeeze latent channel dimension for output
        if output.dim() == 3:
            if output.size(1) == 1:
                output = output.squeeze(1)  # (batch, n_OutputFeatures)
            else:
                output = output.sum(dim=1)  # Sum over latent channels

        return output
