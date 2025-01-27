# LanczosHubbard
Initial implementation of the Lanczos method applied to one dimentional Hubbard model.

# Lanczos Algorithm
The Lanczos Algorithm is an iterative method relevant in the study of ground state of very large Hamiltonians, such as the one of the Hubbard model. This is used to show an example of how these large objects can be studied.

# Hubbard Model
The Hubbard model is a much studied description of interacting electrons in a lattice. Many insteresting characteristics and transitions are thought to naturally arise from the model.

# My Implementation
My implementation includes simple functions for the Lanczos algorithm in 1D, utilizing Spin conservation to evaluate excited states and the likes. A wrapper is included to show a usecase, as well as some plotting function of most relevant quantities

# ToDO:
The code has been written for a class. Intereseting improvements include adding transitions basis, implementing parallelization of processes, including generic observables, better wrapper, better choices of U, support for ground state vector extraction. A possible rewrite could be considered to expand the project to two or more dimensions. 
