import math
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import trig as trig
import data_formatter as darform

def momentum(m, v):
    """Calculates 1D momentum"""
    p = m * v
    return p

def v_momentum(m, v):
    """calculates momentum scalar given a mass and a velocity vector"""
    v_m = np.linalg.norm(v)
    p = momentum(m, v_m)
    return p

def sum_momentum(p_1, p_2):
    """Sum of two input momentums"""
    t_p = p_1 + p_2
    return t_p

def momentum_deviation(p_i, p_f):
    """calculates the deviation of the momentum"""
    p_d = (p_f - p_i) / p_i
    return p_d

def total_momentum(v_1, v_2, m_1, m_2):
    """calculates intial and final momentums"""
    shape_1 = v_1.shape
    l_1 = shape_1[1]
    slices_1 = []
    for i in range(shape_1):
        current_slice = v_1[:, i]
        slices_1.append(np.array(current_slice))

    shape_2 = v_2.shape
    l_2 = shape_2[1]
    slices_2 = []
    for i in range(shape_2):
        current_slice = v_2[:, i]
        slices_2.append(np.array(current_slice))

    p_1_i = v_momentum(m_1, slices_1[0])
    p_1_f = v_momentum(m_1, slices_1[1])
    p_2_i = v_momentum(m_2, slices_2[0])
    p_2_f = v_momentum(m_2, slices_2[1])

    p_i = sum_momentum(p_1_i, p_2_i)
    p_f = sum_momentum(p_1_f, p_2_f)

    return p_i, p_f

def calc_deviation(v_1, v_2, m_1, m_2):
    """calculates total momentum deviation"""
    p_i, p_f = total_momentum(v_1, v_2, m_1, m_2)
    p_d = momentum_deviation(p_i, p_f)

    return p_d