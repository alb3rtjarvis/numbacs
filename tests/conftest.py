import pytest
import numpy as np
import pickle
import sys
sys.path.append('/home/ajarvis/numbacs_git/src')
from numbacs.flows import get_predefined_callable
import os 
from os.path import join

dir_path = os.path.dirname(os.path.realpath(__file__))

#%%
@pytest.fixture
def coords_dg():
    x = np.linspace(0,2,101)
    y = np.linspace(0,1,51)
    
    return x,y

@pytest.fixture
def fm_data():

    return np.load(join(dir_path,"testing_data/fm.npy"))

@pytest.fixture
def fm_n_data():

    return np.load(join(dir_path,"testing_data/fm_n.npy"))

@pytest.fixture
def fm_aux_data():

    return np.load(join(dir_path,"testing_data/fm_aux.npy"))

@pytest.fixture
def ftle_data():

    return np.load(join(dir_path,"testing_data/ftle.npy"))

@pytest.fixture
def C_data():

    return np.load(join(dir_path,"testing_data/C.npy"))

@pytest.fixture
def C_eig_aux_data():

    return (np.load(join(dir_path,"testing_data/Cvals_aux.npy")),
            np.load(join(dir_path,"testing_data/Cvecs_aux.npy")))

@pytest.fixture
def C_eig_data():

    return (np.load(join(dir_path,"testing_data/Cvals.npy")),
            np.load(join(dir_path,"testing_data/Cvecs.npy")))

@pytest.fixture
def lavd_data():

    return np.load(join(dir_path,"testing_data/lavd.npy"))

@pytest.fixture
def ile_data():

    return np.load(join(dir_path,"testing_data/ile.npy"))

@pytest.fixture
def S_data():

    return np.load(join(dir_path,"testing_data/S.npy"))

@pytest.fixture
def S_eig_func_data():

    return (np.load(join(dir_path,"./testing_data/Svals.npy")),
            np.load(join(dir_path,"testing_data/Svecs.npy")))

@pytest.fixture
def S_eig_data():

    return (np.load(join(dir_path,"testing_data/Svals_d.npy")),
            np.load(join(dir_path,"testing_data/Svecs_d.npy")))

@pytest.fixture
def ivd_data():

    return np.load(join(dir_path,"testing_data/ivd.npy"))

@pytest.fixture
def vort_data():

    return np.load(join(dir_path,"testing_data/vort.npy"))

@pytest.fixture
def vel_data():

    return (np.load(join(dir_path,"./testing_data/u.npy")),
            np.load(join(dir_path,"./testing_data/v.npy")))


@pytest.fixture
def flow_callable():
    
    return get_predefined_callable('double_gyre', return_domain=False)

@pytest.fixture
def ridge_pts_data():

    return np.load(join(dir_path,"testing_data/ridge_pts.npy"))

@pytest.fixture
def ridge_data():
    with open(join(dir_path,"/testing_data/ridges.pkl"), 'rb') as f:
        ridges = pickle.load(f)
    return ridges

@pytest.fixture
def ordered_ridge_data():
    with open("./testing_data/ordered_ridges.pkl", 'rb') as f:
        oridges = pickle.load(f)
    return oridges

@pytest.fixture
def hyp_lcs_data():
    with open(join(dir_path,"testing_data/hyplcs.pkl"), 'rb') as f:
        hyp_lcs = pickle.load(f)
    return hyp_lcs

@pytest.fixture
def hyp_oecs_data():
    with open(join(dir_path,"testing_data/hypoecs.pkl"), 'rb') as f:
        hyp_oecs = pickle.load(f)
    return hyp_oecs

@pytest.fixture
def elliptic_lcs_data():
    with open(join(dir_path,"testing_data/elliplcs.pkl"), 'rb') as f:
        elliptic_lcs = pickle.load(f)
    return elliptic_lcs

@pytest.fixture
def elliptic_oecs_data():
    with open(join(dir_path,"testing_data/ellipoecs.pkl"), 'rb') as f:
        elliptic_oecs = pickle.load(f)
    return elliptic_oecs 



