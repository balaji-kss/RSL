import torch.nn as nn
import torch
import torch.nn.init as init
from modelZoo.sparseCoding import *
from utils import *
from modelZoo.actRGB import *
from modelZoo.gumbel_module import *
from scipy.spatial import distance
from modelZoo.transformer import TransformerEncoder, TransformerDecoder

class GroupNorm(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)
    Examples::
        # >>> input = torch.randn(20, 6, 10, 10)
        # >>> # Separate 6 channels into 3 groups
        # >>> m = nn.GroupNorm(3, 6)
        # >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        # >>> m = nn.GroupNorm(6, 6)
        # >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        # >>> m = nn.GroupNorm(1, 6)
        # >>> # Activating the module
        # >>> output = m(input)
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    # __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class binaryCoding(nn.Module):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(161, 64, kernel_size=(3,3), padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(64, 32, kernel_size=(3,3), padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 64, kernel_size=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 500),
            # nn.Linear(64*26*8, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, num_binary)
        )

        for m in self.modules():
            # if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
            #     init.xavier_normal(m.weight.data)
            #     m.bias.data.fill_(0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class binarizeSparseCode(nn.Module):
    def __init__(self, num_binary, Drr, Dtheta, gpu_id, Inference, fistaLam):
        super(binarizeSparseCode, self).__init__()
        self.k = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.fistaLam = fistaLam
        # self.sparsecoding = sparseCodingGenerator(self.Drr, self.Dtheta, self.PRE, self.gpu_id)
        # self.binaryCoding = binaryCoding(num_binary=self.k)
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
        self.binaryCoding = GumbelSigmoid()

    def forward(self, x, T):
        sparseCode, Dict = self.sparseCoding(x, T)
        # sparseCode = sparseCode.permute(2,1,0).unsqueeze(3)
        # # sparseCode = sparseCode.reshape(1, T, 20, 2)
        # binaryCode = self.binaryCoding(sparseCode)

        # reconstruction = torch.matmul(Dict, sparseCode)
        binaryCode = self.binaryCoding(sparseCode, force_hard=True, temperature=0.1, inference=self.Inference)

        # temp = sparseCode*binaryCode
        return binaryCode, sparseCode, Dict

class classificationGlobal(nn.Module):
    def __init__(self, num_class, Npole, dataType):
        super(classificationGlobal, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.dataType = dataType
        self.conv1 = nn.Conv1d(self.Npole, 256, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=256, eps=1e-05, affine=True)

        self.conv2 = nn.Conv1d(256, 512, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=512, eps=1e-5, affine=True)

        self.conv3 = nn.Conv1d(512, 1024, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=1024, eps=1e-5, affine=True)

        self.pool = nn.AvgPool1d(kernel_size=(25))

        self.conv4 = nn.Conv2d(self.Npole + 1024, 1024, (3, 1), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=1024, eps=1e-5, affine=True)

        self.conv5 = nn.Conv2d(1024, 512, (3, 1), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-5, affine=True)

        self.conv6 = nn.Conv2d(512, 256, (3, 3), stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=256, eps=1e-5, affine=True)

        self.fc = nn.Linear(256*10*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.cls = nn.Linear(128, self.num_class)

        self.relu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self,x):
        inp = x
        bz = inp.shape[0]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x_gl = self.pool(self.relu(self.bn3(self.conv3(x))))
        x_new = torch.cat((x_gl.repeat(1,1,inp.shape[-1]),inp),1).reshape(bz,1024+self.Npole,25,2)
        x_out = self.relu(self.bn4(self.conv4(x_new)))
        x_out = self.relu(self.bn5(self.conv5(x_out)))
        x_out = self.relu(self.bn6(self.conv6(x_out)))

        'MLP'
        x_out = x_out.view(bz,-1)  #flatten
        x_out = self.relu(self.fc(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out))

        out = self.cls(x_out)

        return out

class classificationTenc(nn.Module):
    def __init__(self, num_class, Npole, dataType):
        super(classificationTenc, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.dataType = dataType
        self.conv1 = nn.Conv1d(self.Npole, 128, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=128, eps=1e-05, affine=True)

        self.conv2 = nn.Conv1d(128, 128, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=128, eps=1e-5, affine=True)

        self.conv3 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=256, eps=1e-5, affine=True)

        self.pool = nn.AvgPool1d(kernel_size=(25))

        self.conv4 = nn.Conv2d(self.Npole + 256, 256, (3, 1), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=256, eps=1e-5, affine=True)

        self.conv5 = nn.Conv2d(256, 128, (3, 1), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(num_features=128, eps=1e-5, affine=True)

        self.conv6 = nn.Conv2d(128, 128, (3, 3), stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=128, eps=1e-5, affine=True)

        self.fc = nn.Linear(128*10*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.cls = nn.Linear(64, self.num_class)

        self.relu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self,x):

        inp = x
        bz = inp.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x_gl = self.pool(self.relu(self.bn3(self.conv3(x))))

        x_new = torch.cat((x_gl.repeat(1,1,inp.shape[-1]),inp),1).reshape(bz,256+self.Npole,25,2)

        x_out = self.relu(self.bn4(self.conv4(x_new)))
        x_out = self.relu(self.bn5(self.conv5(x_out)))
        x_out = self.relu(self.bn6(self.conv6(x_out)))

        'MLP'
        x_out = x_out.view(bz,-1)  #flatten
        x_out = self.relu(self.fc(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out))

        out = self.cls(x_out)

        return out

class classificationWBinarization(nn.Module):
    def __init__(self, num_class, Npole, num_binary, dataType):
        super(classificationWBinarization, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.dataType = dataType
        self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=Npole,dataType=self.dataType)

    def forward(self, x):
        'x is coefficients'
        inp = x.reshape(x.shape[0], x.shape[1], -1).permute(2,1,0).unsqueeze(-1)
        binaryCode = self.BinaryCoding(inp)
        binaryCode = binaryCode.t().reshape(self.num_binary, x.shape[-2], x.shape[-1]).unsqueeze(0)
        label = self.Classifier(binaryCode)

        return label,binaryCode

class classificationWSparseCode(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta, dataType,dim,fistaLam, gpu_id):
        super(classificationWSparseCode, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.T = T
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=Npole, dataType=self.dataType)
        self.fistaLam = fistaLam
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, lam=self.fistaLam, gpu_id=self.gpu_id)

    def forward(self, x, T):
        sparseCode, Dict, Reconstruction  = self.sparseCoding.forward2(x, T) # w.o. RH

        label = self.Classifier(sparseCode)

        return label, sparseCode, Reconstruction

class Tenc_SparseC_Cl(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta, dataType, dim, fistaLam, gpu_id):
        super(Tenc_SparseC_Cl, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.T = T
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.fistaLam = fistaLam
        
        self.transformer_encoder = TransformerEncoder(embed_dim=25*2, embed_proj_dim=50, is_input_proj=1, ff_dim=2048, num_heads=5, num_layers=2, dropout=0.1)

        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=Npole, dataType=self.dataType)

        self.sparse_coding = DyanEncoder(self.Drr, self.Dtheta, lam=self.fistaLam, gpu_id=self.gpu_id)

    def key_padding_mask(self, lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, x, T, lengths):

        key_padding_mask = ~self.key_padding_mask(lengths, max_len=T).cuda()
        src_mask = None

        tenc_out = self.transformer_encoder(x, src_mask=src_mask, src_key_padding_mask=key_padding_mask)

        sparseCode, Dict, Reconstruction  = self.sparse_coding.forward2(tenc_out, T) # w.o. RH

        label = self.Classifier(sparseCode)

        return label, Reconstruction, tenc_out

class Dyan_Tenc(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta, dataType,dim,fistaLam, gpu_id):
        super(Dyan_Tenc, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.fistaLam = fistaLam
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, lam=self.fistaLam, gpu_id=self.gpu_id)

        self.transformer_encoder = TransformerEncoder(embed_dim=161, embed_proj_dim=None, ff_dim=2048, num_heads=7, num_layers=2, dropout=0.1)

        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(161*50, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.cls = nn.Linear(64, self.num_class)
        self.relu = nn.LeakyReLU()

    def forward(self, x, T):

        sparseCode, Dict, Reconstruction  = self.sparseCoding.forward2(x, T) # w.o. RH

        sparseCode = sparseCode.permute(0, 2, 1) #(B, 161, 50) --> (B, 50, 161)
        tenc_out = self.transformer_encoder(sparseCode, src_mask=None, src_key_padding_mask=None)
        
        tenc_out = self.relu(tenc_out) #(B, 50, 161)
        tenc_out = self.dropout(tenc_out)

        tenc_out = tenc_out.view(tenc_out.shape[0],-1)  #flatten (B, 161 * 50)
        tenc_out = self.relu(self.fc1(tenc_out)) # (B, 1024)
        tenc_out = self.relu(self.fc2(tenc_out)) # (B, 256)
        tenc_out = self.relu(self.fc3(tenc_out)) # (B, 64)
        label = self.cls(tenc_out) # (B, 10)

        return label, sparseCode, Reconstruction

class Tenc_Cl(nn.Module):
    def __init__(self, num_class, gpu_id):
        super(Tenc_Cl, self).__init__()
        self.num_class = num_class
        
        self.transformer_encoder = TransformerEncoder(embed_dim=25*2, embed_proj_dim=25*2, ff_dim=2048, num_heads=5, num_layers=8, dropout=0.1)

        self.fc1 = nn.Linear(36*50, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.cls = nn.Linear(128, self.num_class)
        self.relu = nn.LeakyReLU()

    def key_padding_mask(self, lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, x, T, lengths):

        padding_mask = self.key_padding_mask(lengths, max_len=T)

        tenc_out = self.transformer_encoder(x, padding_mask)

        x_out = tenc_out.view(tenc_out.shape[0],-1)  #flatten
        x_out = self.relu(self.fc1(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out))

        out = self.cls(x_out)

        return out, out, tenc_out

class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)

class Dyan_Autoencoder(nn.Module):
    def __init__(self, Drr, Dtheta, dim, dataType, Inference, gpu_id, fistaLam):
        super(Dyan_Autoencoder, self).__init__()

        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.fistaLam = fistaLam

        print('***** Dyan Autoencoder just mask *****')

        self.transformer_encoder = TransformerEncoder(embed_dim=25*2, embed_proj_dim=50, is_input_proj=1, ff_dim=2048, num_heads=5, num_layers=2, dropout=0.1)
        self.sparse_coding = DyanEncoder(self.Drr, self.Dtheta,  lam=fistaLam, gpu_id=self.gpu_id)
        self.transformer_decoder = TransformerDecoder(embed_dim=25*2, embed_proj_dim=50, is_output_proj=1, ff_dim=2048, num_heads=5, num_layers=2, dropout=0.1)

        self.tgt_mask = self.generate_square_subsequent_mask(36).cuda()

    def key_padding_mask(self, lengths, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def generate_square_subsequent_mask(self, sz: int, device='cpu'):
    
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
    
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def get_src_mask(self, lengths, max_len=None):

        mask = ~torch.arange(0, max_len, device=lengths.device).repeat(max_len, 1).lt(lengths.unsqueeze(1).unsqueeze(1))
        mask = mask.float()
        mask = mask.masked_fill(mask == 1, float('-inf'))

        # EX for size=5, len=2:
        # [[0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf]]

        return mask.repeat(5, 1, 1)

    def get_tgt_mask(self, lengths, max_len=None):

        sqr_subseq_mask = torch.tril(torch.ones(max_len, max_len, device=lengths.device) == 1)
        pad_mask = torch.arange(0, max_len, device=lengths.device).repeat(max_len, 1).lt(lengths.unsqueeze(1).unsqueeze(1))

        mask = pad_mask & sqr_subseq_mask
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        # EX for size=5, len=2:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf]]

        return mask.repeat(5, 1, 1)

    def forward(self, x, T, lengths):
        
        key_padding_mask = ~self.key_padding_mask(lengths, max_len=T).cuda()
        #key_padding_mask = None
        #src_mask = self.get_src_mask(lengths, max_len=T).cuda()
        #tgt_mask = self.get_tgt_mask(lengths, max_len=T).cuda()
        src_mask = None

        tenc_out = self.transformer_encoder(x, src_mask=src_mask, src_key_padding_mask=key_padding_mask)
        #print('tenc_out shape ', tenc_out.shape)

        sparse_code, Dict, _ = self.sparse_coding.forward2(tenc_out, T)
        #print('sparse_code shape ', sparse_code.shape)

        dyan_out = torch.matmul(Dict, sparse_code)
        #print('dyan_out shape ', dyan_out.shape)

        tdec_out = self.transformer_decoder(dyan_out, tgt_mask=self.tgt_mask, tgt_key_padding_mask=key_padding_mask)
        #print('tdec_out shape ', tdec_out.shape)

        return dyan_out, tenc_out, tdec_out

class Fullclassification(nn.Module):
    def __init__(self, num_class, Npole, num_binary, Drr, Dtheta,dim, dataType, Inference, gpu_id, fistaLam):
        super(Fullclassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.fistaLam = fistaLam
        # self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
        self.BinaryCoding = GumbelSigmoid()
        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=Npole, dataType=self.dataType)
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta,  lam=fistaLam, gpu_id=self.gpu_id)
    def forward(self, x, T):
        sparseCode, Dict, R = self.sparseCoding.forward2(x, T) # w.o. RH
        # sparseCode, Dict, Reconstruction  = self.sparseCoding(x, T) # w.RH

        'for GUMBEL'
        binaryCode = self.BinaryCoding(sparseCode**2, force_hard=True, temperature=0.1, inference=self.Inference)
        temp1 = sparseCode * binaryCode
        # temp = binaryCode.reshape(binaryCode.shape[0], self.Npole, int(x.shape[-1]/self.dim), self.dim)
        Reconstruction = torch.matmul(Dict, temp1)

        label = self.Classifier(binaryCode)

        return label, binaryCode, Reconstruction


class twoStreamClassification(nn.Module):
    def __init__(self, num_class, Npole, num_binary, Drr, Dtheta, dim, gpu_id, inference, fistaLam, dataType, kinetics_pretrain):
        super(twoStreamClassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.PRE = PRE
        self.gpu_id = gpu_id
        self.dataType = dataType
        self.dim = dim
        self.kinetics_pretrain = kinetics_pretrain
        self.Inference = inference
        self.fistaLam = fistaLam

        self.dynamicsClassifier = Fullclassification(self.num_class, self.Npole, self.num_binary,
                                self.Drr, self.Dtheta, self.dim, self.dataType, self.Inference, self.gpu_id, self.fistaLam)

        self.RGBClassifier = RGBAction(self.num_class, self.kinetics_pretrain)

    def forward(self,skeleton, image, T, fusion):
        # stream = 'fusion'
        label1, binaryCode, Reconstruction, coeff, dictionary = self.dynamicsClassifier(skeleton, T)
        label2 = self.RGBClassifier(image)

        if fusion:
            label = {'RGB':label1, 'Dynamcis':label2}
        else:
            label = 0.5 * label1 + 0.5 * label2

        # print('dyn:', label1, 'rgb:', label2)

        return label, binaryCode, Reconstruction, coeff, dictionary

if __name__ == '__main__':
    gpu_id = 3
    # net = binaryCoding(num_binary=161).cuda(gpu_id)
    # net = Fullclassification()
    # x1 = torch.randn(20*3,161,1,1).cuda(gpu_id)
    # y1 = net(x1)
    # y1[y1>0] = 1
    # y1[y1<0] = -1
    #
    # x2 = torch.randn(1, 161, 1, 1).cuda(gpu_id)
    # y2 = net(x2)
    # y2[y2>0] = 1
    # y2[y2<0] = -1
    #
    # out_b1 = y1[0].detach().cpu().numpy().tolist()
    # out_b2 = y2[0].detach().cpu().numpy().tolist()
    # dist = distance.hamming(out_b1, out_b2)

    # net = classificationHead(num_class=10, Npole=161).cuda(gpu_id)
    # x = torch.randn(1, 161, 20, 3).cuda(gpu_id)
    #
    # y = net(x)
    N = 2*80
    num_class = 10
    dataType = '2D'

    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    # net = Tenc_SparseC_Cl(num_class=num_class, Npole=N+1, Drr=Drr, Dtheta=Dtheta, dataType=dataType, dim=2, fistaLam=0.1, gpu_id=gpu_id).cuda(gpu_id)
    net = Dyan_Autoencoder(Drr, Dtheta, dim=2, dataType=dataType, Inference=True, gpu_id=gpu_id, fistaLam=0.1).cuda(gpu_id)
    x = torch.randn(1, 36, 50).cuda(gpu_id)
    xImg = torch.randn(1, 20, 3, 224, 224).cuda(gpu_id)
    T = x.shape[1]

    label, _, _ = net(x, T)

    print('check')






