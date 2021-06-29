"Author: Ognyan Petkov"

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.signal import find_peaks as _find_peaks
from scipy import constants
from tkinter import *


def create_spectrum():
    # Retrieving entries #
    temperature = float(T_Entry.get())
    Jmax = int(Jmax_Entry.get())
    Delta = float(Delta_Entry.get())
    ppFWHM = int(ppFWHM_Entry.get())
    SNR = float(SNR_Entry.get())
    upper_constants = [float(U_G_Entry.get()),
                       float(U_B_Entry.get()),
                       float(U_D_Entry.get()),
                       float(U_H_Entry.get()),
                       float(U_L_Entry.get())]
    up_str = ' '.join(str(v) for v in upper_constants)
    lower_constants = [float(L_G_Entry.get()),
                       float(L_B_Entry.get()),
                       float(L_D_Entry.get()),
                       float(L_H_Entry.get()),
                       float(L_L_Entry.get())]
    lower_str = ' '.join(str(v) for v in lower_constants)
    branches = [P_check.get(),
                Q_check.get(),
                R_check.get()]
    spec = Spectrum(temperature, Jmax, Delta, ppFWHM)
    freq, spectrum = spec.generate_spectrum(
        upper_constants, lower_constants, branches
    )
    noisy_spectrum, max_noise = spec.add_noise(SNR)
    peaks, freq_list = spec.find_peaks(noisy_spectrum, max_noise)
    spec.store_spectrum(freq_list, up_str, lower_str)
    spec.plot_spectrum(noisy_spectrum, peaks)


class Spectrum:

    def __init__(self, temperature, Jmax, delta, points_fwhm):
        self.const_k = constants.value(
            'Boltzmann constant in inverse meter per kelvin'
        )
        self.const_k *= 1e-2  # convert to cm^-1 per kelvin
        self.temperature = temperature
        self.Jmax = Jmax
        self.delta = delta
        self.points_fwhm = points_fwhm
        self.Jvec = np.arange(int(self.Jmax))

    def generate_spectrum(self, upper_constants, lower_constants,
                          branches, lineshape='gaussian'):
        upper_energies, lower_energies = self.compute_energy_from_constants(
            upper_constants, lower_constants
        )
        qlines = upper_energies - lower_energies
        plines = np.roll(upper_energies, 1) - lower_energies
        rlines = np.roll(upper_energies, -1) - lower_energies
        plines[0], rlines[-1] = 0., 0.
        lines = np.concatenate((plines, qlines, rlines))
        lines = lines[lines != 0]

        freq_min = np.min(lines) - 10 * self.delta
        freq_max = np.max(lines) + 10 * self.delta
        nspec = int(self.points_fwhm * (freq_max - freq_min) / (self.delta))
        self.freq = np.linspace(freq_min, freq_max, nspec)

        intens = self.calculate_intensity(self.temperature, lower_energies)
        self.spectrum = np.zeros(nspec)

        if lineshape.lower() == 'gaussian':
            for jj in range(0, self.Jmax):
                if branches[0] == 1:
                    self.spectrum += intens[jj] * self.gaussian_profile(plines[jj])
                if branches[1] == 1:
                    self.spectrum += intens[jj] * self.gaussian_profile(qlines[jj])
                if branches[2] == 1:
                    self.spectrum += intens[jj] * self.gaussian_profile(rlines[jj])
        elif lineshape.lower() == 'lorentzian':
            pass
        elif lineshape.lower() == 'voigt':
            pass
        elif lineshape.lower() == 'stick':
            pass
        else:
            pass

        return self.freq, self.spectrum

    def compute_energy_from_constants(self, upper_constants, lower_constants):
        upper_constants = np.array(upper_constants)
        lower_constants = np.array(lower_constants)

        if upper_constants.shape[0] != lower_constants.shape[0]:
            raise SystemExit(
                'The number of upper and lower constants should be the same.'
            )

        nconstants = upper_constants.shape[0]
        # print(nconstants)
        # print(self.Jvec)
        j = np.tile(self.Jvec, (nconstants, 1)).T
        # print(j)
        X = np.power(j * (j + 1), np.arange(nconstants), dtype=np.float)
        # print(X)
        upper_energies = np.dot(X, upper_constants)
        lower_energies = np.dot(X, lower_constants)

        return upper_energies, lower_energies

    def gaussian_profile(self, central_freq):
        return np.exp(-(self.freq - central_freq) ** 2 / (0.36 * self.delta ** 2))

    def stick_spectrum(self, central_freq):
        pass

    def calculate_intensity(self, T, lower_energies):
        gfactor = 2 * self.Jvec + 1
        return gfactor * np.exp(-lower_energies / (self.const_k * T))

    def add_noise(self, snr):
        max_noise = np.max(self.spectrum) / snr
        noise = random.normal(
            loc=0, scale=max_noise, size=(self.spectrum.shape[0])
        )
        noisy_spectrum = self.spectrum + noise

        return noisy_spectrum, max_noise

    def find_peaks(self, spectrum, max_noise):
        peaks, _ = _find_peaks(
            spectrum, height=4.3 * max_noise, distance=self.points_fwhm
        )
        freq_list = [self.freq[peak] for peak in peaks]

        return peaks, freq_list

    def plot_spectrum(self, spectrum, peaks):
        plt.plot(self.freq[peaks], spectrum[peaks], 'x')
        plt.plot(self.freq, spectrum)
        plt.show()

    def store_spectrum(self, freq_list, up, lower):
        with open("spectra.txt", "w") as spectra:
            spectra.write("v' J' v'' J'' Freq ERROR\n")
            for peak in freq_list:
                spectra.write(f"0 0 0 0 {peak} 0.05\n")


# -------GUI------- #
FONT = "Arial"
window = Tk()
window.title("Spectrum generator")
window.config(height=500, width=500)

# -------Labels------- #
Upper_L = Label(text="Upper level G/B/D/H/L:", font=(FONT, 13, "normal"))
Lower_L = Label(text="Lower level G/B/D/H/L:", font=(FONT, 13, "normal"))
Spectra_type_L = Label(text="Spectra: ", font=(FONT, 13, "normal"))
Jmax_L = Label(text="Jmax", font=(FONT, 13, "normal"))
T_L = Label(text="Temperature[K]:", font=(FONT, 13, "normal"))
Delta_L = Label(text="Delta[cm-1]:", font=(FONT, 13, "normal"))
ppFWHM_L = Label(text="ppFWHM", font=(FONT, 13, "normal"))
SNR_L = Label(text="Sound-to-Noise Ratio:", font=(FONT, 13, "normal"))
Upper_L.grid(row=0, column=0)
Lower_L.grid(row=1, column=0)
Spectra_type_L.grid(row=2, column=0)
Jmax_L.grid(row=3, column=0)
T_L.grid(row=4, column=0)
Delta_L.grid(row=5, column=0)
ppFWHM_L.grid(row=6, column=0)
SNR_L.grid(row=7, column=0)

# -------Entries------- #

# Levels
U_G_Entry = Entry(font=(FONT, 13, "normal"), width=10)
U_G_Entry.insert(0, 1100.)
U_B_Entry = Entry(font=(FONT, 13, "normal"), width=10)
U_B_Entry.insert(0, 1.950)
U_D_Entry = Entry(font=(FONT, 13, "normal"), width=10)
U_D_Entry.insert(0, -2.0e-5)
U_H_Entry = Entry(font=(FONT, 13, "normal"), width=10)
U_H_Entry.insert(0, 2.0e-7)
U_L_Entry = Entry(font=(FONT, 13, "normal"), width=10)
U_L_Entry.insert(0, 0)
L_G_Entry = Entry(font=(FONT, 13, "normal"), width=10)
L_G_Entry.insert(0, 100.)
L_B_Entry = Entry(font=(FONT, 13, "normal"), width=10)
L_B_Entry.insert(0, 2.0)
L_D_Entry = Entry(font=(FONT, 13, "normal"), width=10)
L_D_Entry.insert(0, -3.1e-5)
L_H_Entry = Entry(font=(FONT, 13, "normal"), width=10)
L_H_Entry.insert(0, 1.23e-7)
L_L_Entry = Entry(font=(FONT, 13, "normal"), width=10)
L_L_Entry.insert(0, 0)

U_G_Entry.grid(row=0, column=1)
U_B_Entry.grid(row=0, column=2)
U_D_Entry.grid(row=0, column=3)
U_H_Entry.grid(row=0, column=4)
U_L_Entry.grid(row=0, column=5)
L_G_Entry.grid(row=1, column=1)
L_B_Entry.grid(row=1, column=2)
L_D_Entry.grid(row=1, column=3)
L_H_Entry.grid(row=1, column=4)
L_L_Entry.grid(row=1, column=5)

# Type of spectra
P_check = IntVar()
Q_check = IntVar()
R_check = IntVar()

P_Button = Checkbutton(text="P", variable=P_check, onvalue=True, offvalue=False)
Q_Button = Checkbutton(text="Q", variable=Q_check, onvalue=True, offvalue=False)
R_Button = Checkbutton(text="R", variable=R_check, onvalue=True, offvalue=False)
P_Button.grid(row=2, column=1)
Q_Button.grid(row=2, column=2)
R_Button.grid(row=2, column=3)

# Spectra specific

Jmax_Entry = Entry(font=(FONT, 13, "normal"), width=10)
Jmax_Entry.insert(0, 18)
T_Entry = Entry(font=(FONT, 13, "normal"), width=10)
T_Entry.insert(0, 100)
Delta_Entry = Entry(font=(FONT, 13, "normal"), width=10)
Delta_Entry.insert(0, 0.5)
ppFWHM_Entry = Entry(font=(FONT, 13, "normal"), width=10)
ppFWHM_Entry.insert(0, 5)
SNR_Entry = Entry(font=(FONT, 13, "normal"), width=10)
SNR_Entry.insert(0, 50)
Jmax_Entry.grid(row=3, column=1)
T_Entry.grid(row=4, column=1)
Delta_Entry.grid(row=5, column=1)
ppFWHM_Entry.grid(row=6, column=1)
SNR_Entry.grid(row=7, column=1)

# Generate
Generate_Button = Button(text="Generate!", font=(FONT, 13, "normal"), command=create_spectrum)
Generate_Button.grid(row=5, column=3)

window.mainloop()
