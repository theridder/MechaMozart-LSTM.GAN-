import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.autograd import Variable


class DiscriminatorV1(nn.Module):
    """
    Tells the difference between a real piano sonata and a forgery.

    Essentially a GRU, but without a reset gate. Forget and input gates are
    separate, output is not given at every step, but at the very end of
    processing an example.
    Input format:
    Output format:
    Internal State size:
    """

    def __init__(self, note_size=4*10 + 88 + 7, state_size=150):
        super(DiscriminatorV1, self).__init__()
        self.forgetgate() = nn.Linear(state_size + note_size, state_size)
        self.inputgate() = nn.Linear(state_size + note_size, state_size)
        self.candidate_gen() = nn.Linear(note_size, state_size
        self.state() = Variable(torch.zeros(state_size), requires_grad=True)
