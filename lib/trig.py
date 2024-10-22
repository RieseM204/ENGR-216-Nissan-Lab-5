import math
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import lib.phys as phys
import lib.data_formatter as datform

def distance(A : np.ndarray(shape=(2,1)), B : np.ndarray(shape=(2,1))) -> float:
    """Takes two ordered pairs and returns the distance between them"""
    C = np.subtract(A, B)
    r = np.linalg.norm(C)
    return r

def unit(v : np.ndarray(shape=(2,1))) -> np.ndarray(shape=(2,1)): 
    """Takes a 2x1 vector and converts it into a unit vector"""
    l = np.linalg.norm(v)
    v_hat = v / l
    return v_hat



def theta_between(A : np.ndarray(shape=(2,1)), B : np.ndarray(shape=(2,1))) -> float:
    """Takes two 2x1 vectors and returns the angle between them in radians"""
    AB_mag_prod = np.linalg.norm(A) * np.linalg.norm(B)
    AB_dot = np.dot(A.T, B)
    theta = math.acos(AB_dot / AB_mag_prod)
    return theta



def theta_from_posx(A : np.ndarray(shape=(2,1))) -> float:
    """Takes a 2x1 unit vector and returns the internal angle in radians from the positive x-axis"""
    theta = np.arctan2(A[1, 0], A[0, 0])
    return theta



def rotate(A, theta : float) -> np.ndarray(shape=(2,1)):
    """Rotates a 2d vector about the origin counter-clockwise"""
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], #cos(theta) , -sin(theta)
                        [np.sin(theta), np.cos(theta)]]) #sin(theta) ,  cos(theta)
    A_prime = np.dot(rot_mat, A)
    return A_prime



def pythag_inf(A) -> float:
    """takes a list of floats and returns the pythag of all of them"""
    internal = 0
    for i in A:
        internal += float(i ** 2)
    return math.sqrt(internal)