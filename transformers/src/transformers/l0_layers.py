import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import math
import functools

###
# Just a copy from the CVR repository of the L0Linear layer, so that I can import it into the bert model definition
###
class L0Linear(nn.Module):
    """The hard concrete equivalent of ``nn.Linear``.
        Pruning is unstructured, with weights masked at
        the individual parameter level, not neuron level
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init_value: float = 0.0,
        temp: float = 1.,
        l0: bool = False,
        ablate_mask: str = None
    ) -> None:
        """Initialize a L0UstructuredLinear module.

        """
        super().__init__()

        self.l0 = l0
        self.in_features = in_features
        self.out_features = out_features
        self.mask_init_value = mask_init_value
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))  # type: ignore
        self.temp = temp
        self.ablate_mask = ablate_mask

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))  # type: ignore
        else:
            self.register_parameter("bias", None)  # type: ignore

        self.reset_parameters()

        # Create a random tensor to reinit ablated parameters
        if self.ablate_mask == "random":
            self.random_weight = nn.Parameter(torch.zeros(out_features, in_features))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad=False

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        nn.init.constant_(self.mask_weight, self.mask_init_value)

    def get_sampled_mask(self):
        """Used to create a binary mask that contains the same number of ones and zeros as a normal ablated mask,
        but drawn from the complement of the trained binary mask. This is done to assess whether ablating a trained
        subnetwork yields greater performance degredation than ablating a random subnetwork.

        Sample a random mask once and then use it to evaluate a whole dataset. Create more models like this to
        get a distribution over random mask samples
        """
        if hasattr(self, "sampled_mask"):
            return self.sampled_mask
        else:
            sampled_size = torch.sum((self.mask_weight > 0).int())
            if sampled_size > torch.sum((self.mask_weight < 0).int()):
                raise ValueError("Trying to sample random masks, but original mask contains > 50 percent of weights")
            # Sample sample_size number of weights from the complement of the mask given by mask_weight
            complement_mask_weights = self.mask_weight < 0
            sample_complement_indices = complement_mask_weights.nonzero(as_tuple=False) # get indices of complement weights
            # shuffle the indices of possible sampled weights, take the first sample_size indices as your sampled mask 
            sample_complement_indices = sample_complement_indices[torch.randperm(sample_complement_indices.size(0))][:sampled_size]
            # Reformat indices into tuple form for indexing into tensor
            sample_complement_indices = (sample_complement_indices[:, 0], sample_complement_indices[:, 1])
            # Create a mask with just the sampled indices removed to run random ablation experiments
            sampled_mask = torch.ones(self.mask_weight.shape)
            sampled_mask[sample_complement_indices] = 0.0
            self.sampled_mask = nn.Parameter(sampled_mask, requires_grad=False).cuda()
            return self.sampled_mask

    def compute_mask(self):
        if (self.ablate_mask == None) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight > 0).float() # Hard cutoff once frozen or testing
        elif (self.ablate_mask == "sampled"):
            mask = self.get_sampled_mask() # Used to see whether ablating particular mask is worse than a sampled mask
        elif (self.ablate_mask != None) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight <= 0).float() # Used for subnetwork ablation
        else: 
            mask = F.sigmoid(self.temp * self.mask_weight)      
        return mask      

    def train(self, train_bool):
        self.training = train_bool

    def reset_parameters(self):
        """Reset network parameters."""
        if self.l0:
            self.init_mask()

        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # Update Linear reset to match torch 1.12 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


    def forward(self, data: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore
        """Perform the forward pass.
        Parameters
        ----------
        data : torch.Tensor
            N-dimensional tensor, with last dimension `in_features`
        Returns
        -------
        torch.Tensor
            N-dimensional tensor, with last dimension `out_features`
        """
        if self.l0:
            self.mask = self.compute_mask()
            if self.ablate_mask == "random":
                masked_weight = self.weight * self.mask # This will give you the inverse weights, 0's for ablated weights
                masked_weight += (~self.mask.bool()).float() * self.random_weight# Invert the mask to target the 0'd weights, make them random
            else:
                masked_weight = self.weight * self.mask
        else:
            masked_weight = self.weight

        out = F.linear(data, masked_weight, self.bias)
        return out

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        s += ", bias={}".format(str(self.bias is not None))
        return s.format(**self.__dict__)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())