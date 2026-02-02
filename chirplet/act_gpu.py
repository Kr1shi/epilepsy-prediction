import os
import pickle
import cupy as cp
import scipy.integrate as integrate
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import scipy.optimize as optimize
from tqdm import tqdm
import joblib
import psutil
from cupy.cuda import memory
import monitoringclass

class ACT:
    def __init__(
        self,
        FS=256,
        length=3840,
        dict_addr="dict_cache.p",
        tc_info=(0, 3840, 1),
        fc_info=(0.7, 15, 0.2),
        logDt_info=(-4, -1, 0.3),
        c_info=(-30, 30, 3),
        complex=False,
        force_regenerate=False,
        mute=False,
        monitor = False
    ):
        self.FS = FS
        self.length = length
        self.dict_addr = dict_addr
        self.tc_info = tc_info
        self.fc_info = fc_info
        self.logDt_info = logDt_info
        self.c_info = c_info
        self.complex = complex
        self.float32 = True
        
        if os.path.exists(self.dict_addr) and not force_regenerate:
            if not mute:
                print("Found Chirplet Dictionary, Loading File...")
            dict_mat_np, param_mat_np = joblib.load(self.dict_addr)
            self.dict_mat = cp.asarray(dict_mat_np)
            self.param_mat = cp.asarray(param_mat_np)
        else:
            if not mute:
                print("Generating Chirplet Dictionary...")
            if monitor:
                monitoring = monitoringclass.MonitoringClass()
                monitoring.start_CPU_monitoring()
                monitoring.start_GPU_monitoring()
            self.generate_chirplet_dictionary(debug=True)
            joblib.dump((self.dict_mat.get(), self.param_mat.get()), self.dict_addr)
            if monitor:
                monitoring.stop_CPU_monitoring()
                monitoring.stop_GPU_monitoring()
            if not mute:
                print("Dictionary Cached.")

    def g(self, tc=0, fc=1, logDt=0, c=0):
        tc = tc / self.FS
        Dt = cp.exp(logDt)
        t = cp.arange(self.length) / self.FS
        gaussian_window = cp.exp(-0.5 * ((t - tc) / (Dt)) ** 2)
        complex_exp = cp.exp(2j * cp.pi * (c * (t - tc) ** 2 + fc * (t - tc)))
        atom = gaussian_window * complex_exp
        if not self.complex:
            atom = cp.real(atom)

        norm = cp.linalg.norm(atom)
        if norm > 0:
            atom /= norm

        if self.float32:
            atom = atom.astype(cp.float32)

        return atom

    def generate_chirplet_dictionary(self, debug=False):
        tc_vals = cp.arange(self.tc_info[0], self.tc_info[1], self.tc_info[2])
        fc_vals = cp.arange(self.fc_info[0], self.fc_info[1], self.fc_info[2])
        logDt_vals = cp.arange(self.logDt_info[0], self.logDt_info[1], self.logDt_info[2])
        c_vals = cp.arange(self.c_info[0], self.c_info[1], self.c_info[2])

        dict_size = len(tc_vals) * len(fc_vals) * len(logDt_vals) * len(c_vals)
        if debug:
            print("Dictionary length:", dict_size)

        dict_mat = cp.zeros((dict_size, self.length), dtype=cp.float32)
        param_mat = cp.zeros((dict_size, 4), dtype=cp.float32)
        cnt = 0

        slow_cnt = 1  # For debugging purposes (pretty progress marker)

        for tc in tc_vals:
            if debug:
                print("\n{}/{}: \t".format(slow_cnt, len(tc_vals)), end="")
                slow_cnt += 1
            for fc in fc_vals:
                if debug:
                    print(".", end="")
                for logDt in logDt_vals:
                    for c in c_vals:
                        dict_mat[cnt] = self.g(tc=tc, fc=fc, logDt=logDt, c=c)
                        param_mat[cnt] = cp.asarray([tc, fc, logDt, c])
                        cnt += 1

        self.dict_mat = dict_mat
        self.param_mat = param_mat



        return dict_mat, param_mat

    def search_dictionary(self, signal):
        print(signal.shape, self.dict_mat.shape)
        projection_values = self.dict_mat.dot(signal)
        return cp.argmax(projection_values), cp.max(projection_values)

    def transform(self, signal, order=5, debug=False):
        param_list = cp.zeros((order, 4), dtype=cp.float32)
        coeff_list = cp.zeros(order, dtype=cp.float32)
        approx = cp.zeros(len(signal), dtype=cp.float32)
        residue = cp.copy(signal)

        for P in range(order):
            ind, val = self.search_dictionary(residue)
            params = self.param_mat[ind]
            res = optimize.minimize(self.minimize_this, params.get(), args=(residue.get()))
            new_params = cp.array(res.x)
            updated_base_chirplet = self.g(tc=new_params[0], fc=new_params[1], logDt=new_params[2], c=new_params[3])
            updated_chirplet_coeff = updated_base_chirplet.dot(residue)
            new_chirp = updated_base_chirplet * updated_chirplet_coeff
            residue -= new_chirp
            approx += new_chirp
            param_list[P] = new_params
            coeff_list[P] = updated_chirplet_coeff

        return {
            "params": param_list.get(),
            "coeffs": coeff_list.get(),
            "signal": signal.get(),
            "error": cp.sum(residue).get(),
            "residue": residue.get(),
            "approx": approx.get(),
        }
       
    def minimize_this(self, coeffs, signal):
        atom = self.g(tc=coeffs[0], fc=coeffs[1], logDt=coeffs[2], c=coeffs[3])
        return -1 * abs(atom.dot(cp.array(signal)).get())
