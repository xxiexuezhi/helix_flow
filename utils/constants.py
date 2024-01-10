non_standard_to_standard = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'ASX':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA', # Added ASX => ASP
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYL':'LYS', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS', 'SEC':'CYS', # Added pyrrolysine and selenocysteine
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

three_to_one_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'}

one_to_three_letter = {v:k for k,v in three_to_one_letter.items()}

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}

num_to_letter = {v:k for k, v in letter_to_num.items()}

restype_to_heavyatom_names = {
    "ALA": ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "ARG": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    "ASN": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    "ASP": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    "CYS": ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    "GLN": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    "GLU": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    "GLY": ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    "HIS": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    "ILE": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    "LEU": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    "LYS": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    "MET": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    "PHE": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    "PRO": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    "SER": ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    "THR": ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    "TRP": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    "TYR": ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    "VAL": ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    "UNK": ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
}

heavyatom_to_label = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'X': 4} # X is null token

van_der_waals_radius = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "S": 1.8,
}

max_num_heavy_atoms = len(restype_to_heavyatom_names["ALA"])

cg_scheme = {
    "ALA": {
        "1": ["C","CA","CB","N"],
        "2": ["C","CA","O"],
        "3": [],
        "4": [],
    },
    "ARG": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CB", "CG", "CD"],
        "4": ["NE","NH1","NH2","CZ"],
    },
    "ASN": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","ND2","OD1"],
        "4": [],
    },
    "ASP": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","OD1","OD2"],
        "4": [],
    },
    "CYS": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CA","CB","SG"],
        "4": [],
    },
    "GLN": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CD","OE1","NE2"],
        "4": [],
    },
    "GLU": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG", "CD", "OE1", "OE2"],
        "4": [],
    },
    "GLY": {
        "1": ["C", "CA", "N"],
        "2": ["C", "CA", "O"],
        "3": [],
        "4": [],
    },
    "HIS": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CD2","CE1","ND1","NE2"],
        "4": [],
    },
    "ILE": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CB","CG1","CG2"],
        "4": ["CB","CG1","CD1"],
    },
    "LEU": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CD1","CD2"],
        "4": [],
    },
    "LYS": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CB","CG","CD"],
        "4": ["CD","CE","NZ"],
    },
    "MET": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CE","SD"],
        "4": [],
    },
    "PHE": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CD1","CD2","CE1","CE2","CZ"],
        "4": [],
    },
    "PRO": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CB","CG","CD"],
        "4": [],
    },
    "SER": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CA","CB","OG"],
        "4": [],
    },
    "THR": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CB","CG2","OG1"],
        "4": [],
    },
    "TRP": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CD1","CD2","CE2","CE3","CZ2","CZ3","CH2","NE1"],
        "4": [],
    },
    "TYR": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CG","CD1","CD2","CE1","CE2","CZ","OH"],
        "4": [],
    },
    "VAL": {
        "1": ["C", "CA", "CB", "N"],
        "2": ["C", "CA", "O"],
        "3": ["CB","CG1","CG2"],
        "4": [],
    }
}