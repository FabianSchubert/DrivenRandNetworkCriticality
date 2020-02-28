import os

DATA_DIR = os.path.join(os.path.dirname(__file__),"../data")
PLOT_DIR = os.path.join(os.path.dirname(__file__),"../plots")

TEXT_WIDTH = 5.78

BRIGHT_YELLOW = (1.,.8,.0)
BRIGHT_BLUE = (0.,.66,1.)
BRIGHT_GREEN = (0.,1.,.21)
BRIGHT_RED = (1.,0.2,0.1)

MODES = ['heterogeneous_idential_binary_input',
        'heterogeneous_independent_gaussian_input',
        'homogeneous_identical_binary_input',
        'homogeneous_independent_gaussian_input']
