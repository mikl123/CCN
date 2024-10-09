"""
The matrices produced in this module are described in section 4.1 Constraint Module of the paper.
"""

import numpy as np

######
# Note: Since in practice all the datasets had only one stratum, we implemented the code for one stratum
#####


def createIs(file_path, num_classes):
    """
    Generates two matrices, Iplus and Iminus, which correspond to the presence or absence of literals in the constraint bodies.
    """

    # Matrix with indices for positive literals
    Iplus = []
    # Matrix with indeces for negative literals
    Iminus = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            iplus = np.zeros(num_classes)
            iminus = np.zeros(num_classes)
            for item in split_line[3:]:
                if 'n' in item:
                    index = int(item[1:])
                    iminus[index] = 1
                else:
                    index = int(item)
                    iplus[index] = 1
            Iplus.append(iplus)
            Iminus.append(iminus)
    Iplus = np.array(Iplus)
    Iminus = np.array(Iminus)
    return Iplus, Iminus


def createM(file_path, num_classes):
    """
    Matrix M maps the head of each constraint to a binary representation.
    Each row corresponds to a specific constraint with its associated target class.
    """
    M = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            assert split_line[2] == ':-'
            m = np.zeros(num_classes)
            index = int(split_line[1])
            m[index] = 1
            M.append(m)
    M = np.array(M).transpose()
    return M
