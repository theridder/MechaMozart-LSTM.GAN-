import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.autograd import Variable


class DiscriminatorV1(nn.Module):
    """
    Tells the difference between a real piano sonata and a forgery.

    Essentially a GRU, but without a reset gate. Forget and input gates are
    separate. Output is not given at every step, but at the very end of
    processing a track.
    -Input format: x is a 2D tensor. First axis lenght is amount of notes in
    the example. Each column along second axis consists of 4*10 one-hot arrays,
    i.e. a four digit number representing milliseconds since previous note;
    another 4*10 representing lenght of note; 128 entries representing which
    note is being activated, and 7 more to represent velocity 1-127.
    -Output format: a 1D tensor with two entries, representing probablilities
    of real/fake.
    """

    def __init__(self, note_size=4*10 + 4*10 + 128 + 7, state_size=200):
        super(DiscriminatorV1, self).__init__()
        self.forgetgate = nn.Linear(state_size + note_size, state_size)
        self.inputgate = nn.Linear(state_size + note_size, state_size)
        self.candidate_gen = nn.Linear(note_size, state_size)
        self.state = Variable(torch.zeros(state_size), requires_grad=True)

        self.outputlayer1 = nn.Linear(state_size, state_size)
        self.outputlayer2 = nn.Linear(state_size, state_size)
        self.outputlayer3 = nn.Linear(state_size, 2)

    def forward(self, track):
        for note in track:
            gate_input = torch.cat(note, self.state)

            forget = F.sigmoid(self.forgetgate(gate_input))
            self.state *= forget

            inp = F.sigmoid(self.inputgate(gate_input))
            candidates = F.tanh(self.cadidate_gen(note))
            self.state += inp * candidates

        output = F.relu(self.outputlayer1(self.state))
        output = F.relu(self.outputlayer2(output))
        return F.log_softmax(output)
