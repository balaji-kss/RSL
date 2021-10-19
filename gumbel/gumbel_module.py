import torch
import torch.nn as nn


class HardSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):

        y_hard = input.clone()
        y_hard = y_hard.zero_()
        y_hard[input >= 0.7] = 1

        return y_hard

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GumbelSigmoid(torch.nn.Module):
    def __init__(self):
        """
        Implementation of gumbel softmax for a binary case using gumbel sigmoid.
        """
        super(GumbelSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumbel_samples_tensor = -torch.log(
            eps - torch.log(uniform_samples_tensor + eps)
        )

        return gumbel_samples_tensor

    def gumbel_sigmoid_sample(self, logits, temperature, inference=False):
        """Adds noise to the logits and takes the sigmoid. No Gumbel noise during inference."""

        if not inference:
            gumbel_samples_tensor = self.sample_gumbel_like(logits.data)
            gumbel_trick_log_prob_samples = logits + gumbel_samples_tensor.data
        else:
            gumbel_trick_log_prob_samples = logits
            
        soft_samples = self.sigmoid(gumbel_trick_log_prob_samples / temperature)

        return soft_samples

    def gumbel_sigmoid(self, logits, temperature=1, hard=False, inference=False):
        out = self.gumbel_sigmoid_sample(logits, temperature, inference)
        if hard:
            out = HardSoftmax.apply(out)

        return out

    def forward(self, logits, force_hard=False, temperature=2 / 3):
        inference = not self.training

        if self.training and not force_hard:
            return self.gumbel_sigmoid(
                logits, temperature=temperature, hard=False, inference=inference
            )
        else:
            return self.gumbel_sigmoid(
                logits, temperature=temperature, hard=True, inference=inference
            )
