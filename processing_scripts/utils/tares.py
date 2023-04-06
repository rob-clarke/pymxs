import os,sys
sys.path.append(os.path.join(os.path.expanduser('~/projects/baerig/BAERigUI')))

# from api.experiment import Experiment
# from api.experiment.tares import Tare

from . import load_dir

def create_tare(data):
    expr = Experiment("",data).summary_df()
    return Tare.from_interps("",expr,"pitch")

def create_tare_from_dir(dirpath):
    tare_data = load_dir(dirpath)
    return create_tare(tare_data)
