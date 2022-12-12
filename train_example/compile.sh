#!/bin/bash

set -ex

for i in `seq 0 3`; do python pt2jpt.py model_wb97m_d.yml pc14i_v2_wb97m_gas_cl.sae wb97m_gas_ens${i}_cl3_re.pt wb97m_gas_ens${i}.jpt; done
python mk_ens.py 0 wb97m_gas_ens?.jpt wb97m_gas_ens.jpt wb97m_gas_ens_f.jpt

for i in `seq 0 3`; do python pt2jpt.py model_wb97m_d_nb.yml pc14i_v2_wb97m_gas_cl.sae wb97m_gas_ens${i}_cl3_re.pt wb97m_gas_ens${i}_nb.jpt; done
python mk_ens.py 1 wb97m_gas_ens?_nb.jpt wb97m_gas_ens_nb.jpt wb97m_gas_ens_nb_f.jpt

