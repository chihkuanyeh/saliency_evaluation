# use the following args to obtain results on local explanations
args = {
    'seed': 0,
    'sen_r': 0.2,
    'sen_N': 50,  # set to 50 for the experiments used in the paper
    'sg_r': 0.2,
    'sg_N': 500,
    'model': 'models/madry_nat_tf_weight.npz',
    # 'perts': ['Square'],
    'perts': ['Gaussian'],
    # 'exps': ['Grad', 'Int_Grad', 'GBP', 'NB'],
    # 'sgs': ['Grad', 'Int_Grad', 'GBP']
}

# use the following args to obtain results on global explanations
'''
args = {
    'seed': 0,
    'sen_r': 0.1,
    'sen_N': 50,  # set to 50 for the experiments used in the paper
    'sg_r': 0.2,
    'sg_N': 50,
    'model': 'models/madry_nat_tf_weight.npz',
    'perts': ['Square'],
    'exps': ['SHAP', 'Square', 'Grad', 'Int_Grad', 'GBP'],
    'sgs': ['Grad', 'Int_Grad', 'GBP']
}
'''
