"""
@Filename: origin_fractions.py

@Author: Thomas Wagenhäuser, IAU
@Date:   2022-02-18T15:26:24+01:00
@Email:  wagenhaeuser@iau.uni-frankfurt.de

"""


import pandas as pd
import numpy as np
import pathlib
from pathlib import Path
import os
import fnmatch


def find_file(pattern, path):
    """Find all files whose name match a pattern.

    Paramaters:
    ----------
    pattern : string
        name that you are looking for e.g. '*.dat'
    path : string or pathlib.Path
        path to target folder

    Return:
    ---------
    list of absolute filepaths
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(pathlib.Path(root) / name)
    return result


# %%
class EXTR_fraction(object):
    """Object for calculating extra tropical origin fractions from delta Theta and equivalent latitude.

    Use a two dimensional empiric parameterization curve to mathematically calculate
    the extra tropical origin fraction for a given location in the stratosphere for a
    given season. Location must be provided in equivalent latitude - delta theta space.
    If no time information is provided, then the annual mean parameterization is used.

    The empirical mathematical formulation is given as python synthax at
    > def twoD_gumbel_gauss_offset_gumbel_add_twoD_gauss

    The empirical fit parameters for each hemisphere and season were calculated
    based on the original CLaMS model data, published by Hauck et al. 2020,
    using least square fitting methods.

    Usage:
    --------------
    # create EXTR_fraction object in order to load fraction parameters from .csv files:
    In [1]: calc_fracts = EXTR_fraction()

    # calculate fraction, by using EXTR_fraction object like a function:
    In [2]: calc_fracts(-58.3, 14.8, month=8)

    # calling the EXTR_fraction object like in [2]: is equivalent to calling
    In [3]: calc_fracts.calculate_exTR_fraction(-58.3, 14.8, month=8)

    Parameters for the calculation:
    ------------------------------
    eqlat: np.array of length N or float or int
        equivalent latitude data. If neither NH_flag nor SH_flag are provided, then
        eqlat will be used to choose from Norhtern Hemisphere or Southern Hemisphere
        and select corresponing fraction parameters.
    dtheta: np.array of length N or float or int
        delta theta altitude data (potential temperature relative to tropopause)
    datetime: datetimeindex of length N or datetime object, optional
        time of observation. Is used to choose the corresponding seasonal fraction
        parameters. Either datetime, month, seas_i, seas_n or annual must be provided.
        The default is None.
    month: np.array of length N or int, optional
        month of observation. Is used to choose the corresponding seasonal fraction
        parameters. Either datetime, month, seas_i, seas_n or annual must be provided.
        The default is None.
    seas_i: np.array of length N or int, optional
        season of observation. 0: annual mean. 1: DJF. 2: MAM. 3: JJA. 4: SON.
        Is used to choose the corresponding seasonal fraction parameters.
        Either datetime, month, seas_i, seas_n or annual must be provided.
        The default is None.
    seas_n: np.array of length N or str, optional
        string representation of season of observation. Choose from ANN, DJF, MAM, JJA and SON.
        Is used to choose the corresponding seasonal fraction parameters.
        Either datetime, month, seas_i, seas_n or annual must be provided.
        The default is None.
    annual: bool, optional
        if True, then dateim or month input will be ignored and instead the annual mean
        fit parameters will be used. The default is False.
    NH_flag: bool np.array of length N or bool, optional
        will be used instead of eqlat to select corresponing fraction parameters.
        The default is None.
    SH_flag: bool np.array of length N or bool, optional
        will be used instead of eqlat to select corresponing fraction parameters.
        If NH_flag is provided, then SH_flag will be ignored (redundand information).
        The default is None.

    """

    def __init__(self, folder=None):
        self.folder = folder
        self.params, self.popt_paths = self.load_params()
        self.param_names = self.get_param_names()

    def __call__(self, *args, **kwargs):
        return self.calculate_exTR_fraction(*args, **kwargs)

    def load_params(self):
        if self.folder is None:
            self.folder = Path(__file__).parent
        popt_paths = find_file("*F*_*_p12fit.csv", self.folder)
        popt = []
        stem = []
        seas_n = []
        seas_i = []
        hemi = []
        for _path in popt_paths:
            popt.append(np.loadtxt(_path, delimiter=","))
            _stem = _path.stem
            _F, _seas, _end = _stem.split("_")
            hemi.append(_F[0] + "H")
            seas_i.append(int(_F[2]))
            seas_n.append(_seas)
            stem.append(_path.stem)
        df = pd.DataFrame(
            {
                "popt": popt,
                "stem": stem,
                "seas_n": seas_n,
                "seas_i": seas_i,
                "hemi": hemi,
            }
        )
        return df, popt_paths

    def get_param_names(self):
        try:
            with open(self.popt_paths[0], "r") as f:
                f.readline()
                line2 = f.readline()
            return line2.strip().split(",")
        except Exception as E:
            print("Warning in EXTR_fraction.get_param_names:")
            print(E)
            print(
                "Setting parameter names to 'x0,x1,y0,y1,by,e0,e1,gy0,gy1,ga,gx0,gx1'."
            )
            return [
                "x0",
                "x1",
                "y0",
                "y1",
                "by",
                "e0",
                "e1",
                "gy0",
                "gy1",
                "ga",
                "gx0",
                "gx1",
            ]

    @staticmethod
    def _check_single_input(var):
        if isinstance(var, int) or isinstance(var, float):
            single_input = True
        else:
            single_input = False
        return single_input

    @staticmethod
    def _assign_to_hemisphere(eqlat, NH_flag, SH_flag):
        eqlat = np.asarray(eqlat, dtype=float).flatten()
        if (NH_flag is None) and (SH_flag is None):
            NH_flag = eqlat >= 0
            SH_flag = ~NH_flag
        elif NH_flag is not None:
            NH_flag = np.asarray(NH_flag).flatten()
            SH_flag = ~NH_flag
        elif SH_flag is not None:
            SH_flag = np.asarray(SH_flag).flatten()
            NH_flag = ~SH_flag
        hemi = np.asarray(NH_flag, dtype=str)
        hemi[NH_flag] = "NH"
        hemi[SH_flag] = "SH"
        return eqlat, NH_flag, SH_flag, hemi

    @staticmethod
    def _mask_below_tropopause(dtheta):
        dtheta = np.asarray(dtheta, dtype=float).flatten()
        dtheta[dtheta < 0] = np.nan
        return dtheta

    @staticmethod
    def _assign_seas_identifier(datetime, month, seas_i, seas_n, annual):
        seas = np.asarray(0).flatten()
        if (datetime is not None) or (month is not None):
            if datetime is not None:
                month = datetime.month
            month = np.asarray(month, dtype=float).flatten()
            if annual:
                seas = np.zeros(month.shape)
            else:
                seas = month * np.nan
                seas[(month >= 12) | (month < 3)] = 1
                seas[(month >= 3) & (month < 6)] = 2
                seas[(month >= 6) & (month < 9)] = 3
                seas[(month >= 9) & (month < 12)] = 4
        if seas_i is not None:
            seas = np.asarray(seas_i).flatten()
        if seas_n is not None:
            seas_n = np.asarray(seas_n).flatten()
            s_names = {"ANN": 0, "DJF": 1, "MAM": 2, "JJA": 3, "SON": 4}
            seas = np.asarray([s_names[_seas_i] for _seas_i in seas_n])
        if len(seas) == 1:
            seas = seas[0]
        return seas

    def _handle_input_calculate_exTR_fraction(
        self, eqlat, dtheta, datetime, month, seas_i, seas_n, annual, NH_flag, SH_flag
    ):
        # handle single input
        self.single_input = self._check_single_input(eqlat)

        # assign data to NH or SH
        eqlat, NH_flag, SH_flag, hemi = self._assign_to_hemisphere(
            eqlat, NH_flag, SH_flag
        )

        # get rid of data below tropopause
        dtheta = self._mask_below_tropopause(dtheta)

        # assign standardized season identifier to data
        seas = self._assign_seas_identifier(datetime, month, seas_i, seas_n, annual)

        # create DataFrame
        try:
            df = pd.DataFrame(
                {
                    "eqlat": eqlat,
                    "NH flag": NH_flag,
                    "SH flag": SH_flag,
                    "hemi": hemi,
                    "dtheta": dtheta,
                    "seas_i": seas,
                }
            )
        except ValueError as VE:
            print("Error in EXTR_fraction._handle_input_calculate_exTR_fraction:")
            print(VE)
            print("len(eqlat): {}".format(len(eqlat)))
            print("len(NH_flag): {}".format(len(NH_flag)))
            print("len(SH_flag): {}".format(len(SH_flag)))
            print("len(hemi): {}".format(len(hemi)))
            print("len(dtheta): {}".format(len(dtheta)))
            print("len(seas): {}".format(len(seas)))
            raise VE

        # merge loaded parameters with input data
        self.df = df
        dfm = df.merge(self.params, on=["seas_i", "hemi"], how="left")
        dfm[self.param_names] = pd.DataFrame(dfm["popt"].to_list(), index=dfm.index)
        return dfm

    def calculate_exTR_fraction(
        self,
        eqlat,
        dtheta,
        datetime=None,
        month=None,
        seas_i=None,
        seas_n=None,
        annual=False,
        NH_flag=None,
        SH_flag=None,
        min_fract_exTR=0,
        max_fract_exTR=1,
    ):
        # create standardized DataFrame from input:
        # assign NH or SH flags, season identifier
        dfm = self._handle_input_calculate_exTR_fraction(
            eqlat, dtheta, datetime, month, seas_i, seas_n, annual, NH_flag, SH_flag
        )

        # calculate fractions
        dfm["fract_exTR"] = np.nan
        dfm["fract_exTR"] = self.twoD_gumbel_gauss_offset_gumbel_add_twoD_gauss(
            (dfm["eqlat"], dfm["dtheta"]), *dfm[self.param_names].values.T
        )

        # apply maximum and minimum fraction
        dfm.loc[dfm["fract_exTR"] < min_fract_exTR] = min_fract_exTR
        dfm.loc[dfm["fract_exTR"] > max_fract_exTR] = max_fract_exTR

        # store function results
        self.dfm = dfm

        # return fractions
        result = dfm["fract_exTR"].values
        if self.single_input:
            result = result[0]
        return result

    @staticmethod
    def gauss(x, x0, x1):
        return np.exp(-(((x - x1) / x0) ** 2))

    @staticmethod
    def cumgumbel(x, x0, x1):
        return np.exp(-np.exp(-(x - x0) / x1))

    @staticmethod
    def gauss2D(x_data_tuple, x0, x1, y0, y1, a=1):
        X, Y = x_data_tuple
        return a * EXTR_fraction.gauss(X, x0, x1) * EXTR_fraction.gauss(Y, y0, y1)

    @staticmethod
    def cumgumbelXgaussY(x_data_tuple, x0, x1, y0, y1):
        X, Y = x_data_tuple
        return EXTR_fraction.cumgumbel(np.abs(X), x0, x1) * EXTR_fraction.gauss(
            Y, y0, y1
        )

    @staticmethod
    def twoD_gumbel_gauss_offset_gumbel_add_twoD_gauss(
        x_data_tuple, x0, x1, y0, y1, by, e0, e1, gy0=1, gy1=1, ga=0, gx0=1, gx1=1
    ):
        X, Y = x_data_tuple
        peak1 = EXTR_fraction.cumgumbelXgaussY(x_data_tuple, x0, x1, y0, y1)
        peak2 = EXTR_fraction.gauss2D(x_data_tuple, gx0, gx1, gy0, gy1, ga)
        offset_gumbel = EXTR_fraction.cumgumbel(Y, e0, e1)
        Z = peak1 + peak2 + by * offset_gumbel
        return Z.ravel()
