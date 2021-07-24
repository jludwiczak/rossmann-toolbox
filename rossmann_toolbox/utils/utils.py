import numpy as np
from scipy.signal import find_peaks


def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f'{filename}:{lineno} - {message}')


def postprocess_crf(tags, probs):
    _, peaks = find_peaks(tags, width=30)
    peak_dict = {i: (beg, end, np.max(probs[beg:end])) for i, (beg, end) in
                 enumerate(zip(peaks['left_bases'], peaks['right_bases']))}
    return peak_dict


def corr_seq(seq):
    """
    Corrects sequence by mapping non-std residues to 'X'
    :param seq: input sequence
    :return: corrected sequence with non-std residues changed to 'X'
    """
    letters = set(list('ACDEFGHIKLMNPQRSTVWYX'))
    seq = ''.join([aa if aa in letters else 'X' for aa in seq])
    return seq


def separate_beta_helix(secondary):
    '''
    changes labels of helices in dssp sequences, adding number for them for instance:
    `H-E1-H-E2-H` to `'H1-E1-H2-E2-H3'
    :params iterable with chars indicating dssp annotations
    "return listo of chars enhanced dssp annotations"
    '''
    if isinstance(secondary, str):
        secondary = list(secondary)
    elif not isinstance(secondary, Iterable):
        raise ValueError(f'secondary must be iterable, but passed {type(secondary)}')
        
    sec_len = len(secondary)
    #E1 E2 split condition
    h_start = sec_len//2
    #H split condition
    e_indices = [i for i, letter in enumerate(secondary)  if letter =='E']
    e_min, e_max = min(e_indices), max(e_indices)
    secondary_extended = list()
    for i, letter in enumerate(secondary):
        if letter == 'E':
            if i <= h_start:
                new_letter = 'E1'
            else:
                new_letter = 'E2'
        elif letter == 'H':
            if i < e_min:
                new_letter = 'H1'
            elif e_min < i < e_max:
                new_letter = 'H2'
            else:
                new_letter = 'H3'
        elif letter == ' ':
            new_letter = '-'
        else:
            new_letter = letter
        secondary_extended.append(new_letter)
    return secondary_extended


