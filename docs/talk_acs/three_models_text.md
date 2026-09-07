# Slide text — Three models, one prediction

Every measurement is one row of 2,130 columns: 2,048 Morgan fingerprint bits
(substructures the extractant contains), 64 process conditions (acid,
diluent, modifier, concentrations, temperature, contact time), 10 RDKit
descriptors (the extractant's size, polarity and flexibility), 4 metal
columns (which lanthanide, and its ionic radius) and 4 complex-plan counts
(how the complex was assembled around the metal).

Each model predicts log D for one measurement. A separation is the
difference between two predictions inside the same block.

The three are combined with weights fitted on the separations, using only
the extractants not being scored.
