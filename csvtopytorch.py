import torch
from torch.autograd import Variable
from os import listdir


def csv_to_variable(lines):
    # Get ticks per part (quarter note) from header
    tpp = int(lines[0].split()[-1])
    # Get tempo (always close to the start, so loop will break early)
    for line in lines:
        if "Tempo" in line:
            tempo = int(line.split()[-1])
            break
    else:
        raise Exception("No tempo found")

    # Tempo is in microseconds per part, tpp in ticks per part
    # Given x in ticks, this will return y in milliseconds
    def t_to_ms(x):
        return round(x * tempo / (1000 * tpp))

    for index, line in list(enumerate(lines))[3:-2]:
        elements = line.split()
        time = int(elements[1][:-1])

        

# for filename in listdir("./csv_clean"):
#     if filename[0] == ".":
#         continue
