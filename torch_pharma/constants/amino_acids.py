# amino acids
aas = [
    ('G', 'GLY'), ('A', 'ALA'), ('V', 'VAL'), ('L', 'LEU'),
    ('I', 'ILE'), ('F', 'PHE'), ('W', 'TRP'), ('Y', 'TYR'),
    ('D', 'ASP'), ('H', 'HIS'), ('N', 'ASN'), ('E', 'GLU'),
    ('K', 'LYS'), ('Q', 'GLN'), ('M', 'MET'), ('R', 'ARG'),
    ('S', 'SER'), ('T', 'THR'), ('C', 'CYS'), ('P', 'PRO')
]


# amino acid smiles
aa_smiles = {
    'G': 'C(C(=O)O)N',
    'A': 'O=C(O)C(N)C',
    'V': 'CC(C)[C@@H](C(=O)O)N',
    'L': 'CC(C)C[C@@H](C(=O)O)N',
    'I': 'CC[C@H](C)[C@@H](C(=O)O)N',
    'F': 'NC(C(=O)O)Cc1ccccc1',
    'W': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
    'Y': 'N[C@@H](Cc1ccc(O)cc1)C(O)=O',
    'D': 'O=C(O)CC(N)C(=O)O',
    'H': 'O=C([C@H](CC1=CNC=N1)N)O',
    'N': 'NC(=O)CC(N)C(=O)O',
    'E': 'OC(=O)CCC(N)C(=O)O',
    'K': 'NCCCC(N)C(=O)O',
    'Q': 'O=C(N)CCC(N)C(=O)O',
    'M': 'CSCC[C@H](N)C(=O)O',
    'R': 'NC(=N)NCCCC(N)C(=O)O',
    'S': 'C([C@@H](C(=O)O)N)O',
    'T': 'C[C@H]([C@@H](C(=O)O)N)O',
    'C': 'C([C@@H](C(=O)O)N)S',
    'P': 'OC(=O)C1CCCN1'
}
