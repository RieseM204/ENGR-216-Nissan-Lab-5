import math
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import lib.phys as phys
import lib.trig as trig
import lib.data_formatter as datform

df = pd.read_csv("bin/trail_1.csv")

print(df)