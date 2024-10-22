import math
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import phys as phys
import trig as trig

def cvs_to_df(n):
    df = pd.read_csv(f"bin/trail_{n}.csv")
    return df