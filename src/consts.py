import torch.nn as nn
from src.loss import MSEWithPairAttentionLoss

DOS_INTERPOLATION_GRID_PTS = 200
# DOS_ENERGY_WINDOW_LOW = -4.0 #value for DOSCARs
# DOS_ENERGY_WINDOW_HIGH = 4.0
# DOS_ENERGY_WINDOW_LOW = -2.0
# DOS_ENERGY_WINDOW_HIGH = 2.0

PT_COLUMN_NAME = "Feature_PT6_div3000"

# TODO: add spetial loss for wrong sign prediction
#CRITERION = nn.MSELoss()
#CRITERION = MSEWithPairAttentionLoss(pair_weight=1.0)

STANDARD_CONST_DOS = 10
# STANDARD_CONST_TARGET = 10


POSSIBLE_TARGETS = ["MAE", "ESOC_bottom", "ESOC_top", "ESOC", "MAE+ESOC", "DORBITALS", "MAE+ESOC+DORBITALS"]
