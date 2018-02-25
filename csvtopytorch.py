import torch
from torch.autograd import Variable
# from os import listdir
import sys


def csv_to_variable(lines):
    """
    This takes a 'cleaned' csv file, as output by the csvclean.py script.
    It converts it to a pytorch variable, in the format defined in
    Discriminator.py.
    """

    # Get ticks per part (quarter note) from header
    tpp = int(lines[0].split()[-1])
    tempo = 60 / tpp
    time_index = 0
    track = []

    for index, line in list(enumerate(lines))[2:-2]:
        if "Tempo" in line:
            tempo = int(line.split()[-1])
            # Tempo is in microseconds per part, tpp in ticks per part
            # Given x in ticks, this will return y in milliseconds
            conv_factor = tempo / (1000 * tpp)
            continue

        elements = line.split()
        vel = int(elements[-1])
        # 0 velocity means this is actually a note turn off event
        if vel == 0:
            continue
        ticks = int(elements[1][:-1])
        note = int(elements[-2][:-1])

        # Velocity is in a 7 bit binary representation
        velocity = list(map(int, list(bin(vel)[2:])))
        velocity = [0] * (7 - len(velocity)) + velocity

        # Time since previous note is stored in a 4*10 one-hot array
        # 0 to 9999 milliseconds.
        t_since_prev = [0] * 40
        tsp = round((ticks - time_index) * conv_factor)
        tsp = min(9999, tsp)
        tsp = [int(d) for d in str(tsp)]
        tsp = [0] * (4 - len(tsp)) + tsp
        for i, digit in enumerate(tsp):
            t_since_prev[i*10 + digit] = 1

        # This keeps track of where in the track we are
        time_index = ticks

        # A one-hot array representing which note is being played
        note_array = [0] * 128
        note_array[note] = 1

        # Lenght of note is stored similarly as time since previous note
        # We have to look through the next few lines to see where it finishes
        for next_line in lines[index:]:
            if "Note" not in next_line:
                continue

            info = next_line.split()
            if info[-1] == "0" and int(info[-2][:-1]) == note:
                end_tick = int(info[1][:-1])
                break
        else:
            raise Exception("Note at {} never finishes.".format(time_index))

        length = [0] * 40
        l = end_tick - ticks
        l = min(9999, l)
        l = [int(d) for d in str(l)]
        l = [0] * (4 - len(l)) + l
        for index, digit in enumerate(l):
            length[index*10 + digit] = 1

        track.append(t_since_prev + length + note_array + velocity)

    return Variable(torch.Tensor(track))


in_lines = sys.stdin.readlines()
print(csv_to_variable(in_lines))
# for filename in listdir("./csv_clean"):
#     if filename[0] == ".":
#         continue
