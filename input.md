# Input types and shapes

## `NB_MODE == 0` : Dense mode.

Required keys: 
    coord: (B, N, 3)
    numbers: (B, N)
    charge: (B, )
    mult: (B, )

Padding: numbers == 0    

Coulomb modes: simple, dsf

Outputs:
    _natom (1,) if no padding input else (B,)


## `NB_MODE == 1` : Semi-dense mode


Required_keys: 
    coord: (N, 3)
    numbers: (N, )
    charge: (M, )
    mult: (M, )
    nbmat: (N, K)
    mol_idx: (N, ) if M > 1 else not required
    nbmat_lr: (N, L) if model.lr == True

Optional keys:
    cell: (M, 3, 3)
    shifts: (N, K, 3) 

Padding:
    nbmat == numbers.shape[0]

Coulomb modes: simple (if no PBC), DSF, Ewald


## `NB_MODE == 2` : Batched semi-dense mode

Required_keys: 
    coord: (B, N, 3)
    numbers: (N, N)
    charge: (B, )
    mult: (B, )
    nbmat: (B, N, K)
    nbmat_lr: (B, N, L) if model.lr == True

Optional keys:
    cell: (B, 3, 3)
    shifts: (B, K, 3) 

Padding: nbmat == numbers.shape[0]

Coulomb modes: simple (if no PBC), DSF




