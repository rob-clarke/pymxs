import os,sys
sys.path.append(os.path.join(os.getcwd(),'../../baerig/BAERigUI'))

from api.experiment import Experiment
from api.experiment.tares import Tare

import utils

def create_tare(data):
    expr = Experiment("",data).summary_df()
    return Tare.from_interps("",expr,"pitch")

def create_tare_from_dir(dirpath):
    tare_data = utils.load_dir(dirpath)
    return utils.tares.create_tare(tare_data)
