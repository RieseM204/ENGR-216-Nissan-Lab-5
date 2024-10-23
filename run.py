import math
import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import lib.phys as phys
import lib.trig as trig
import lib.data_formatter as datform

"""Main file to be run"""

def main() -> None:
    """Main function"""
    dev_list = []
    k_dev_list = []
    ang_list = []

    for n in range(1, 41):
        df = pd.read_csv(f"bin/trial_{n}.csv")
        df.drop(df.columns[20:], axis=1, inplace=True)
        df = df.dropna(axis='index', how='any')

        DO_m = 27.9
        G_m = 28.4

        DO_x = df['position_px_x-darkorange'].tolist()
        DO_y = df['position_px_y-darkorange'].tolist()
        G_x = df['position_px_x-green'].tolist()
        G_y = df['position_px_y-green'].tolist()

        DO_vx = df['vx-darkorange'].tolist()
        DO_vy = df['vy-darkorange'].tolist()
        G_vx = df['vx-green'].tolist()
        G_vy = df['vy-green'].tolist()

        dist_list = []

        try:
            for i in range(len(DO_x)):
                DO = np.array([[DO_x[i]],[DO_y[i]]])
                G = np.array([[G_x[i]],[G_y[i]]])
                dist = trig.distance(DO, G)
                dist_list.append(float(dist))

            col = dist_list.index(min(dist_list)) 
            DO_vx_i = DO_vx[col-1]
            DO_vy_i = DO_vy[col-1]

            G_vx_i = G_vx[col-1]
            G_vy_i = G_vy[col-1]

            DO_vx_f = DO_vx[col+1]
            DO_vy_f = DO_vy[col+1]

            G_vx_f = G_vx[col+1]
            G_vy_f = G_vy[col+1]

            DO_vi = np.array([[DO_vx_i],[DO_vy_i]])
            DO_vf = np.array([[DO_vx_f],[DO_vy_f]])
            G_vi = np.array([[G_vx_i],[G_vy_i]])
            G_vf = np.array([[G_vx_f],[G_vy_f]])

            DO_v = np.hstack((DO_vi, DO_vf))
            G_v = np.hstack((G_vi, G_vf))

            col_ang = trig.theta_between(DO_vi, G_vi)
            p_dev = phys.calc_deviation(DO_v, G_v, DO_m, G_m)
            k_dev = phys.calc_KE_d(DO_v, G_v, DO_m, G_m)
            ang_list.append(float(col_ang))
            dev_list.append(float(p_dev))
            k_dev_list.append(float(k_dev))
        except (IndexError, ValueError, TypeError):
            print(f"Invalid Trial: {n}")
            pass

    ang_list = [i * 180/math.pi for i in ang_list]

    plt.scatter(ang_list, dev_list, label='Deviation')

    coefficients = np.polyfit(ang_list, dev_list, 1)
    trend_line = np.polyval(coefficients, ang_list)
    plt.plot(ang_list, trend_line, color='red', label="Trend Line")
    residuals = dev_list - trend_line
    std_dev = np.std(residuals)
    print(f"std p = {std_dev}")
    plt.fill_between(ang_list, trend_line - std_dev, trend_line + std_dev, color='gray', alpha=0.3, label="±1 Std. Dev.")
    plt.axvline(x=90, linestyle = '--', color = 'green', label='90 Degrees')
    plt.axhline(y=0, linestyle = '--', color = 'black', label='y = 0')

    plt.ylabel('Momentum Deviation')
    plt.xlabel('Angle (deg)')
    plt.title('Deviation of Momentum by Collision Angle')
    plt.legend()
    plt.show()


    plt.scatter(ang_list, k_dev_list, label='Deviation')

    coefficients = np.polyfit(ang_list, k_dev_list, 1)
    trend_line = np.polyval(coefficients, ang_list)
    plt.plot(ang_list, trend_line, color='red', label="Trend Line")
    residuals = k_dev_list - trend_line
    std_dev = np.std(residuals)
    print(f"std k = {std_dev}")
    plt.fill_between(ang_list, trend_line - std_dev, trend_line + std_dev, color='gray', alpha=0.3, label="±1 Std. Dev.")
    plt.axvline(x=90, linestyle = '--', color = 'green', label='90 Degrees')
    plt.axhline(y=0, linestyle = '--', color = 'black', label='y = 0')

    plt.ylabel('KE Deviation')
    plt.xlabel('Angle (deg)')
    plt.title('Deviation of Kinetic Energy by Collision Angle')
    plt.legend()
    plt.show()

    plt.hist(dev_list, bins = int(math.sqrt(len(dev_list))))
    plt.xlabel('Part Deviation of Momentum')
    plt.ylabel('n')
    plt.title('Histogram of Deviations from Conservation of Momentum')
    plt.show()
    plt.hist(k_dev_list, bins = int(math.sqrt(len(k_dev_list))))
    plt.xlabel('Part Change in Total KE')
    plt.ylabel('n')
    plt.title('Histogram of Changes in Total Kinetic Energy')
    plt.show()

if __name__ == "__main__":
    main()