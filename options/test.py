from rxnmapper import RXNMapper

rxn_mapper = RXNMapper()
rxns = [
    "CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F",
    "C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12",
]
results = rxn_mapper.get_attention_guided_atom_maps(rxns)
print(results)