import pandas as pd
import numpy as np


class ADI:

    def __init__(self):
        self.df = pd.DataFrame()
        self.prev_adi_value = None

    def info(self):
        info = ("The name accumulation/distribution comes from the idea that during accumulation buyers are in control and the price will be bid up through the day,"
                " or will make a recovery if sold down, in either case more often finishing near the day's high than the low. The opposite applies during distribution. ")
        return info

    def __get_ad_util(self, clv_vol):
        self.prev_adi_value += clv_vol
        return self.prev_adi_value

    def get_value_df(self, df):
        df["CLV"] = ((df["close"] - df["low"]) - (df["high"] -
                                                  df["close"])) / (df["high"] - df["low"])
        df["CLV_VOL"] = df["CLV"] * df["volume"]
        self.prev_adi_value = df["CLV_VOL"][0]
        df["ADI"] = df["CLV_VOL"].apply(self.__get_ad_util)
        df = df.drop(["CLV", "CLV_VOL"], axis=1)
        return df

    def get_value_list(self, high_values, low_values, close_values, volume_values):

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values
        })
        self.df["CLV"] = ((self.df["close"] - self.df["low"]) - (self.df["high"] -
                                                                 self.df["close"])) / (self.df["high"] - self.df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * self.df["volume"]
        self.prev_adi_value = self.df["CLV_VOL"][0]
        ad_values = self.df["CLV_VOL"].apply(self.__get_ad_util)

        self.df = pd.DataFrame(None)
        return ad_values


class ATR:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Average True Range is a volatility indicator which provides degree of price of volatility making use of smoothed moving average of true ranges.")
        return info

    def get_value_df(self, df, time_period=14):
        df["close_prev"] = df["close"].shift(1)
        df["TR"] = df[["high", "close_prev"]].max(
            axis=1) - df[["low", "close_prev"]].min(axis=1)
        df["ATR"] = df["TR"].rolling(window=time_period).mean()

        df = df.drop(["close_prev", "TR"], axis=1)

        return df

    def get_value_list(self, high_values, low_values, close_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })

        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["TR"] = self.df[["high", "close_prev"]].max(
            axis=1) - self.df[["low", "close_prev"]].min(axis=1)
        atr_values = self.df["TR"].rolling(window=time_period).mean()

        self.df = pd.DataFrame(None)

        return atr_values


class CMF:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Chaikin Money flow is used to measure money flow volume over a certain time periods.")
        return info

    def get_value_df(self, df, time_period=20):
        df["CLV"] = (2 * df["close"] - (df["high"] +
                                        df["low"])) / (df["high"] - df["low"])
        df["CLV_VOL"] = df["CLV"] * df["volume"]
        df["CLV_VOL_SUM"] = df["CLV_VOL"].rolling(
            window=time_period).sum()
        df["VOL_SUM"] = df["volume"].rolling(
            window=time_period).sum()

        df["CMF"] = df["CLV_VOL_SUM"] / df["VOL_SUM"]

        df = df.drop(["CLV", "CLV_VOL", "CLV_VOL_SUM", "VOL_SUM"], axis=1)

        return df

    def get_value_list(self, high_values, low_values,  close_values, volume_values, time_period=20):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values
        })

        self.df["CLV"] = (2 * self.df["close"] - (self.df["high"] +
                                                  self.df["low"])) / (self.df["high"] - self.df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * self.df["volume"]
        self.df["CLV_VOL_SUM"] = self.df["CLV_VOL"].rolling(
            window=time_period).sum()
        self.df["VOL_SUM"] = self.df["volume"].rolling(
            window=time_period).sum()

        cmf_values = self.df["CLV_VOL_SUM"] / self.df["VOL_SUM"]
        self.df = pd.DataFrame(None)

        return cmf_values


class CHO:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Chaikin oscillator is designed to anticipate the directional changes in Accumulation "
                "distributin line by measuring the momentum behind the movements.")
        return info

    def get_value_df(self, df, short_time_period=3, long_time_period=10):
        df["AD"] = ((2 * df["close"] - (df["high"] + df["low"])
                     ) / (df["high"] - df["low"])) * df["volume"]
        df["AD_short"] = df["AD"].ewm(span=short_time_period).mean()
        df["AD_long"] = df["AD"].ewm(span=long_time_period).mean()

        df["CHO"] = df["AD_short"] - df["AD_long"]
        df = df.drop(["AD", "AD_short", "AD_long"], axis=1)
        return df

    def get_value_list(self, high_values, low_values, close_values, volume_values, short_time_period=3, long_time_period=10):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values
        })
        self.df["AD"] = ((2 * self.df["close"] - (self.df["high"] + self.df["low"])
                          ) / (self.df["high"] - self.df["low"])) * self.df["volume"]
        self.df["AD_short"] = self.df["AD"].ewm(span=short_time_period).mean()
        self.df["AD_long"] = self.df["AD"].ewm(span=long_time_period).mean()

        cho_values = self.df["AD_short"] - self.df["AD_long"]

        self.df = pd.DataFrame(None)
        return cho_values


class CHV:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Chaikin Volatility determines the volatility of instrument using percentage change in a moving average of difference "
                "between high price and the low price over a specific period of time.")
        return info

    def get_value_df(self, df, time_period=10):
        df["difference"] = df["high"] - df["low"]
        df["difference_EMA"] = df["difference"].ewm(
            span=time_period).mean()
        df["difference_EMA_n_periods_ago"] = df["difference_EMA"].shift(
            time_period)
        df["CHV"] = (df["difference_EMA"] - df["difference_EMA_n_periods_ago"]
                     ) / df["difference_EMA_n_periods_ago"] * 100

        df["CHV"].iloc[:time_period] = np.nan

        df = df.drop(["difference", "difference_EMA",
                      "difference_EMA_n_periods_ago"], axis=1)

        return df

    def get_value_list(self, high_values, low_values, time_period=10):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values
        })

        self.df["difference"] = self.df["high"] - self.df["low"]
        self.df["difference_EMA"] = self.df["difference"].ewm(
            span=time_period).mean()
        self.df["difference_EMA_n_periods_ago"] = self.df["difference_EMA"].shift(
            time_period)
        chv_values = (self.df["difference_EMA"] - self.df["difference_EMA_n_periods_ago"]
                      ) / self.df["difference_EMA_n_periods_ago"] * 100
        self.df = pd.DataFrame(None)
        return chv_values


class DPO:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Detrend Price Oscillator tries to eliminates long term trends in order to easily identify small term trends")
        return info

    def get_value_df(self, df, time_period=20):
        df["close_prev"] = df["close"].shift(int(time_period / 2 + 1))
        df["SMA"] = df["close"].rolling(window=time_period).mean()

        df["DPO"] = df["close_prev"] - df["SMA"]

        df = df.drop(["close_prev", "SMA"], axis=1)
        return df

    def get_value_list(self, close_values, time_period=20):
        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(
            int(time_period / 2 + 1))
        self.df["SMA"] = self.df["close"].rolling(window=time_period).mean()

        dpo_values = self.df["close_prev"] - self.df["SMA"]
        self.df = pd.DataFrame(None)
        return dpo_values


class EMV:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Ease of movement tries to identify amount of volume needed to move prices.")
        return info

    def get_value_df(self, df, volume_divisor=1000000, need_moving_average=True, time_period=14):
        df["H+L"] = df["high"] + df["low"]
        df["H+L_prev"] = df["H+L"].shift(1)
        df["MIDPT"] = (df["H+L"] / 2 - df["H+L_prev"] / 2)
        df["BOXRATIO"] = ((df["volume"] / volume_divisor) /
                          (df["high"] - df["low"]))
        df["EMV"] = (df["MIDPT"] / df["BOXRATIO"]) * 100

        if need_moving_average:
            df["EMV"] = df["EMV"].rolling(window=time_period).mean()

        df = df.drop(["H+L", "H+L_prev", "MIDPT", "BOXRATIO"], axis=1)

        return df

    def get_value_list(self, high_values, low_values, volume_values, volume_divisor=1000000, need_moving_average=True, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "volume": volume_values
        })
        self.df["H+L"] = self.df["high"] + self.df["low"]
        self.df["H+L_prev"] = self.df["H+L"].shift(1)
        self.df["MIDPT"] = (self.df["H+L"] / 2 - self.df["H+L_prev"] / 2)
        self.df["BOXRATIO"] = (
            (self.df["volume"] / volume_divisor) / (self.df["high"] - self.df["low"]))

        self.df["EMV"] = self.df["MIDPT"] / self.df["BOXRATIO"] * 100
        if need_moving_average:
            self.df["EMV"] = self.df["EMV"].rolling(window=time_period).mean()
        emv_values = self.df["EMV"]
        self.df = pd.DataFrame(None)
        return emv_values


class EMA:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Exponential Moving Average is a type of moving average which puts more weightage to the"
                "recent points, where as moving average puts same weightage all the points in consideration")
        return info

    def get_value_df(self, df, close_col=None, time_period=21):
        if close_col is not None:
            if close_col < len(df.columns):
                df["EMA"] = df.iloc[:, close_col].ewm(span=time_period).mean()
            else:
                # throw exception
                pass
        else:
            if "close" in df.columns:
                df["EMA"] = df["close"].ewm(span=time_period).mean()
            elif "Close" in df.columns:
                df["EMA"] = df["Close"].ewm(span=time_period).mean()
            elif "CLOSE" in df.columns:
                df["EMA"] = df["CLOSE"].ewm(span=time_period).mean()
            else:
                # throw exception
                pass

    def get_value_list(self, close_values, time_period=21):
        self.df["close"] = close_values
        ema_values = self.df["ema"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)
        return ema_values


class FI:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Force index tries to determine the amount of power used to move the price of an asset")
        return info

    def get_value_df(self, df, time_period=14):
        df["close_prev"] = df["close"].shift(1)
        df["FI"] = (df["close"] -
                    df["close_prev"]) * df["volume"]
        df["FI"] = df["fi"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)
        df = df.drop(["close_prev"], axis=1)
        return df

    def get_value_list(self, close_values, volume_values, time_period=14):
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["FI"] = (self.df["close"] -
                         self.df["close_prev"]) * self.df["volume"]
        fi_values = self.df["FI"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)
        return fi_values


class MI:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Mass index tries to determine the range of high and low values over a specified period of time")
        return info

    def get_value_df(self, df, time_period=25, ema_time_period=9):
        df["difference"] = df["high"] - df["low"]
        df["difference_EMA"] = df["difference"].ewm(
            span=ema_time_period).mean()
        df["difference_double_EMA"] = df["difference_EMA"].ewm(
            span=ema_time_period).mean()
        df["MI"] = df["difference_EMA"] / \
            df["difference_double_EMA"]
        df["MI"] = df["MI"].rolling(window=time_period).sum()
        df = df.drop(["difference", "difference_EMA",
                      "difference_double_EMA"], axis=1)

    def get_value_list(self, high_values, low_values, time_period=25, ema_time_period=9):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values
        })
        self.df["difference"] = self.df["high"] - self.df["low"]
        self.df["difference_EMA"] = self.df["difference"].ewm(
            span=ema_time_period).mean()
        self.df["difference_double_EMA"] = self.df["difference_EMA"].ewm(
            span=ema_time_period).mean()
        self.df["MI"] = self.df["difference_EMA"] / \
            self.df["difference_double_EMA"]
        self.df["MI"] = self.df["MI"].rolling(window=time_period).sum()

        mi_values = self.df["MI"]
        self.df = pd.DataFrame(None)
        return mi_values


class MED:
    def __init__(self):
        self.df = None

    def info(self):
        info = (
            "Median determines the mid point of the price range of a particular time period")
        return info

    def get_value_df(self, df):
        df["MED"] = (df["high"] + df["low"]) / 2
        return df

    def get_value_list(self, high_values, low_values):
        return (high_values + low_values) / 2


class MOM:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Momentum helps to determine the price changes from one period to another.")
        return info

    def get_value_df(self, df, time_period=1):
        df["close_prev"] = df["close"].shift(time_period)
        df["MOM"] = df["close"] - df["close_prev"]
        self.df = pd.DataFrame(None)

    def get_value_list(self, close_values, time_period=1):
        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(time_period)
        self.df["MOM"] = self.df["close"] - self.df["close_prev"]
        mom_values = self.df["MOM"]
        self.df = pd.DataFrame(None)
        return mom_values


class MFI:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Money flow index uses price and volume data to for identifying overbought and oversold signals of an asset")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["TP"] = (df["low"] + df["high"] +
                         df["close"]) / 3
        self.df["TP_prev"] = self.df["TP"].shift(1)
        self.df["PORN"] = np.zeros(len(df))
        self.df.loc[self.df["TP"] > self.df["TP_prev"], "PORN"] = np.float(1)
        df["RMF"] = df["TP"] * df["volume"]
        df["NMF"], df["PMF"] = np.zeros(len(df)), np.zeros(len(df))
        df.loc[df["PORN"] == 0.0, "NMF"] = df["RMF"]
        df.loc[df["PORN"] == 1.0, "PMF"] = df["RMF"]

        df["NMF"] = df["NMF"].rolling(window=time_period).sum()
        df["PMF"] = df["PMF"].rolling(window=time_period).sum()
        df["MFI_ratio"] = df["PMF"] / (df["NMF"] + 0.0000001)
        df["MFI"] = (100 - (100 / (1 + df["MFI_ratio"])))

    def get_value_list(self, high_values, low_values, close_values, volume_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values
        })
        self.df["TP"] = (self.df["low"] + self.df["high"] +
                         self.df["close"]) / 3
        self.df["TP_prev"] = self.df["TP"].shift(1)
        self.df["PORN"] = np.zeros(len(self.df))
        self.df.loc[self.df["TP"] > self.df["TP_prev"], "PORN"] = np.float(1)

        self.df["RMF"] = self.df["TP"] * self.df["volume"]
        self.df["NMF"], self.df["PMF"] = np.zeros(
            len(self.df)), np.zeros(len(self.df))
        self.df.loc[self.df["PORN"] == 0.0, "NMF"] = self.df["RMF"]
        self.df.loc[df["PORN"] == 1.0, "PMF"] = self.df["RMF"]

        self.df["NMF"] = self.df["NMF"].rolling(window=time_period).sum()
        self.df["PMF"] = self.df["PMF"].rolling(window=time_period).sum()
        self.df["MFI_ratio"] = self.df["PMF"] / (self.df["NMF"] + 0.0000001)
        mfi_values = (100 - (100 / (1 + self.df["MFI_ratio"])))

        self.df = None
        return mfi_values


class MACD:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Moving Average Convergence is a trend following momentum indicator that "
                "shows a relationship between two moving averages of an asset")
        return info

    def get_value_df(self, df, short_time_period=12, long_time_period=26, need_signal=True, signal_time_period=9):
        df["LONG_EMA"] = df["close"].ewm(span=long_time_period).mean()
        df["SHORT_EMA"] = df["close"].ewm(span=short_time_period).mean()

        df["MACD"] = df["SHORT_EMA"] - df["LONG_EMA"]
        if need_signal:
            df["MACD_signal_line"] = df["MACD"].ewm(
                span=signal_time_period).mean()

        df = df.drop(["LONG_EMA", "SHORT_EMA"], axis=1)
        return df

    def get_value_list(self, close_values, short_time_period=12, long_time_period=26, need_signal=True, signal_time_period=9):
        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["LONG_EMA"] = self.df["close"].ewm(
            span=long_time_period).mean()
        self.df["SHORT_EMA"] = self.df["close"].ewm(
            span=short_time_period).mean()

        self.df["MACD"] = self.df["SHORT_EMA"] - self.df["LONG_EMA"]
        self.df["MACD_signal_line"] = self.df["MACD"].ewm(
            span=signal_time_period).mean()

        macd_values, macd_signal_line_values = self.df["MACD"], self.df["MACD_signal_line"]
        self.df = pd.DataFrame(None)
        if need_signal:
            return macd_values, macd_signal_line_values
        return macd_values


class NegativeDirectionIndicator:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Negative Direction Indicator is a component of Average Directional Index "
                "and provides a signal that whether downtrend is increasing.")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["DM-"] = np.zeros(len(df))
        self.df["low_prev"] = df["low"].shift(1)
        self.df["high_prev"] = df["high"].shift(1)

        self.df.loc[(self.df["low_prev"]-df["low"]) > (df["high"] -
                                                       self.df["high_prev"]), "DM-"] = self.df["low_prev"] - df["low"]

        self.df["DM-smoothed"] = self.df["DM-"].rolling(
            window=time_period).sum()

        if "ATR" not in self.df.columns:
            AverageTrueRange().get_value_df(self.df)

        df["DI-"] = self.df["DM-smoothd"] / self.df["ATR"]
        self.df = pd.DataFrame(None)

    def get_value_df(self, high_values, low_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values
        })

        self.df["DM-"] = np.zeros(len(self.df))
        self.df["low_prev"] = self.df["low"].shift(1)
        self.df["high_prev"] = self.df["high"].shift(1)

        self.df.loc[(self.df["low_prev"]-self.df["low"]) > (self.df["high"] -
                                                            self.df["high_prev"]), "DM-"] = self.df["low_prev"] - self.df["low"]

        self.df["DM-smoothed"] = self.df["DM-"].rolling(
            window=time_period).sum()

        if "ATR" not in self.df.columns:
            AverageTrueRange().get_value_df(self.df)

        di_minus_values = self.df["DM-smoothd"] / self.df["ATR"]

        self.df = pd.DataFrame(None)
        return di_minus_values


class NVI:
    def __init__(self):
        self.df = pd.DataFrame(None)
        self.prev_nvi_value = 1000

    def info(self):
        info = ("Negative Volume Index helps in identifying trends and reversals.")
        return info

    def __get_nvi_util(self, df_row):
        if df_row["volume"] < df_row["volume_prev"]:
            self.prev_nvi_value = self.prev_nvi_value * \
                (1 + ((df_row["close"] - df_row["close_prev"]) /
                      df_row["close_prev"]))
        return self.prev_nvi_value

    def get_value_df(self, df, start_value=1000):
        self.prev_nvi_value = start_value

        df["close_prev"] = df["close"].shift(1)
        df["volume_prev"] = df["volume"].shift(1)

        df["NVI"] = df.apply(self.__get_nvi_util, axis=1)

        df = df.drop(["close_prev", "volume_prev"], axis=1)

        return df

    def get_value_list(self, close_values, volume_values, start_value=1000):
        self.prev_nvi_value = start_value
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })

        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["volume_prev"] = self.df["volume"].shift(1)

        nvi_values = self.df.apply(self.__get_nvi_util, axis=1)

        self.df = pd.DataFrame(None)
        return nvi_values


class OBV:
    def __init__(self):
        self.df = None
        self.obv_value = None

    def info(self):
        info = ("On Balance Volume provides the signal whether the volume is flowing in or out of a given security.")
        return info

    def __get_obv_util(self, df_row):
        if df_row["close"] > df_row["close_prev"]:
            self.obv_value = self.obv_value + df_row["volume"]
        elif df_row["close"] < df_row["close_prev"]:
            self.obv_value = self.obv_value - df_row["volume"]
        return self.obv_value

    def get_value_df(self, df):

        self.obv_value = df.iloc[0]["volume"]
        df["close_prev"] = df["close"].shift(1)
        df["OBV"] = df.apply(self.__get_obv_util, axis=1)

        df = df.drop(["close_prev"], axis=1)
        return df

    def get_value_list(self, close_values, volume_values):
        self.obv_value = df.iloc[0]["volume"]
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })

        self.df["close_prev"] = self.df["close"].shift(1)
        obv_values = self.df.apply(self.__get_obv_util, axis=1)

        return obv_values


class PositiveDirectionIndicator:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Positive Direction Indicator is a component of Average Directional Index "
                "and provides a signal that whether the uptrend is increasing or not")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["DM+"] = np.zeros(len(df))
        self.df["low_prev"] = df["low"].shift(1)
        self.df["high_prev"] = df["high"].shift(1)

        self.df.loc[(self.df["low_prev"]-df["low"]) < (df["high"] -
                                                       self.df["high_prev"]), "DM-"] = df["high"] - self.df["high_prev"]

        self.df["DM+smoothed"] = self.df["DM+"].rolling(
            window=time_period).sum()

        if "ATR" not in self.df.columns:
            AverageTrueRange().get_value_df(self.df)

        df["DI+"] = self.df["DM+smoothed"] / self.df["ATR"]
        self.df = pd.DataFrame(None)

    def get_value_list(self, high_values, low_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values
        })
        self.df["DM+"] = np.zeros(len(self.df))
        self.df["low_prev"] = self.df["low"].shift(1)
        self.df["high_prev"] = self.df["high"].shift(1)

        self.df.loc[(self.df["low_prev"]-self.df["low"]) < (self.df["high"] -
                                                            self.df["high_prev"]), "DM-"] = self.df["high"] - self.df["high_prev"]

        self.df["DM+smoothed"] = self.df["DM+"].rolling(
            window=time_period).sum()

        if "ATR" not in self.df.columns:
            AverageTrueRange().get_value_df(self.df)

        df_plus_values = self.df["DM+smoothed"] / self.df["ATR"]

        self.df = pd.DataFrame(None)
        return df_plus_values


class PositiveVolumeIndex:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Negative Volume Index helps in identifying trends and reversals.")
        return info

    def get_value_df(self, df, starting_value=100):
        pvi_values = [starting_value]

        for i in range(1, len(df)):
            if df.iloc[i]["volume"] <= df.iloc[i-1]["volume"]:
                pvi_values.append(pvi_values[i-1])
            else:
                pvi_values.append(
                    (1 + ((df.iloc[i]["close"] - df.iloc[i-1]["close"]) / df.iloc[i-1]["close"])) * pvi_values[i-1])
        df["PVI"] = pvi_values

    def get_value_list(self, close_values, volume_values, starting_value=100):
        pvi_values = [starting_value]
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })
        for i in range(1, len(df)):
            if self.df.iloc[i]["volume"] <= self.df.iloc[i-1]["volume"]:
                pvi_values.append(pvi_values[i-1])
            else:
                pvi_values.append(
                    (1 + ((self.df.iloc[i]["close"] - self.df.iloc[i-1]["close"]) / self.df.iloc[i-1]["close"])) * pvi_values[i-1])

        self.df = pd.DataFrame(None)
        return pvi_values


class PVT:
    def __init__(self):
        self.df = None
        self.pvt_value = None

    def info(self):
        info = ("Price Volume Trend helps in identifying trend by using cumulative volume adjusted by change in price")
        return info

    def __get_pvt_util(self, df_row):
        if not np.isnan(df_row["close"]) and not np.isnan(df_row["close_prev"]):
            self.pvt_value = self.pvt_value + \
                (((df_row["close"] - df_row["close_prev"]) /
                  df_row["close_prev"]) * df_row["volume"])
        return self.pvt_value

    def get_value_df(self, df):
        self.pvt_value = df.iloc[0]["volume"]
        df["close_prev"] = df["close"].shift(1)
        df["PVT2"] = df.apply(self.__get_pvt_util, axis=1)

    def get_value_list(self, close_values, volume_values):
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })
        self.pvt_value = self.df.iloc[0]["volume"]
        self.df["close_prev"] = self.df["close"].shift(1)
        pvt_values = self.df.apply(self.__get_pvt_util, axis=1)

        self.df = pd.DataFrame(None)
        return pvt_values


class PC:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Price channels forms a boundary and between them the close price of an asset is oscillating")
        return info

    def get_value_df(self, df, percent_value=6, ema_period=21):
        df["EMA_FOR_PC"] = df["close"].ewm(span=ema_period).mean()

        df["PC_upper"] = df["EMA_FOR_PC"] * (1 + (percent_value / 100))
        df["PC_lower"] = df["EMA_FOR_PC"] * (1 - (percent_value / 100))

        df = df.drop(["EMA_FOR_PC"], axis=1)

        return df

    def get_value_list(self, close_values, percent_value=6, ema_period=21):
        self.df = pd.DataFrame({
            "close": close_values
        })

        self.df["EMA_FOR_PC"] = self.df["close"].ewm(span=ema_period).mean()

        pc_upper_values = self.df["EMA_FOR_PC"] * (1 + (percent_value / 100))
        pc_lower_values = self.df["EMA_FOR_PC"] * (1 - (percent_value / 100))

        self.df = pd.DataFrame(None)
        return pc_upper_values, pc_lower_values


class PO:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Price oscillator is a momentum osciallator which shows a difference between two moving averages")
        return info

    def get_value_df(self, df, short_ema_period=9, long_ema_period=26):
        df["Short_EMA"] = df["close"].ewm(
            span=short_ema_period).mean()

        df["Short_EMA"].iloc[:short_ema_period] = np.nan

        df["Long_EMA"] = df["close"].ewm(span=long_ema_period).mean()
        df["Long_EMA"].iloc[:long_ema_period] = np.nan

        df["PO"] = ((df["Short_EMA"] - df["Long_EMA"]) /
                    df["Long_EMA"]) * 100
        df = df.drop(["Short_EMA", "Long_EMA"], axis=1)
        return df

    def get_value_list(self, close_values, short_ema_period=9, long_ema_period=26):
        self.df = pd.DataFrame({
            "close": close_values
        })

        self.df["Short_EMA"] = self.df["close"].ewm(
            span=short_ema_period).mean()
        self.df["Short_EMA"].iloc[:short_ema_period] = np.nan

        self.df["Long_EMA"] = self.df["close"].ewm(span=long_ema_period).mean()
        self.df["Long_EMA"].iloc[:long_ema_period] = np.nan

        po_values = ((self.df["Short_EMA"] - self.df["Long_EMA"]) /
                     self.df["Long_EMA"]) * 100
        self.df = pd.DataFrame(None)

        return po_values


class ROC:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Rate of change helps in calculation of speed of ascent or descent.")
        return info

    def get_value_df(self, df, time_period=12):
        df["close_prev"] = df["close"].shift(time_period)

        df["ROC"] = (df["close"] - df["close_prev"]) / \
            df["close_prev"] * 100

        df = df.drop(["close_prev"], axis=1)

        return df

    def get_value_list(self, close_values, time_period=12):
        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["close_prev"] = self.df["close"].shift(time_period)

        roc_values = (self.df["close"] - self.df["close_prev"]) / \
            self.df["close_prev"] * 100
        self.df = pd.DataFrame(None)
        return roc_values


class RSI:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Relative Strength Index is used to generate oversold and overbought signals.")
        return info

    def get_value_df(self, df, time_period=14):
        df["close_prev"] = df["close"].shift(1)

        df["GAIN"] = 0.0
        df["LOSS"] = 0.0

        df.loc[df["close"] > df["close_prev"],
               "GAIN"] = df["close"] - df["close_prev"]
        df.loc[df["close_prev"] > df["close"],
               "LOSS"] = df["close_prev"] - df["close"]
        df["AVG_GAIN"] = df["GAIN"].ewm(span=time_period).mean()
        df["AVG_LOSS"] = df["LOSS"].ewm(span=time_period).mean()
        df["AVG_GAIN"].iloc[:time_period] = np.nan
        df["AVG_LOSS"].iloc[:time_period] = np.nan
        df["RS"] = df["AVG_GAIN"] / \
            (df["AVG_LOSS"] + 0.00000001)  # to avoid divide by zero

        df["RSI"] = 100 - ((100 / (1 + df["RS"])))

        df = df.drop(["close_prev", "GAIN", "LOSS",
                      "AVG_GAIN", "AVG_LOSS", "RS"], axis=1)

        self.df = pd.DataFrame(None)

        return df

    def get_value_list(self, close_values, time_period=14):

        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["GAIN"] = 0.0
        self.df["LOSS"] = 0.0

        self.df.loc[df["close"] > self.df["close_prev"],
                    "GAIN"] = self.df["close"] - self.df["close_prev"]

        self.df.loc[self.df["close_prev"] > self.df["close"],
                    "LOSS"] = self.df["close_prev"] - self.df["close"]

        self.df["AVG_GAIN"] = self.df["GAIN"].ewm(span=time_period).mean()
        self.df["AVG_LOSS"] = self.df["LOSS"].ewm(span=time_period).mean()
        self.df["AVG_GAIN"].iloc[:time_period] = np.nan
        self.df["AVG_LOSS"].iloc[:time_period] = np.nan

        self.df["RS"] = self.df["AVG_GAIN"] / \
            (self.df["AVG_LOSS"] + 0.000001)  # to avoid divide by zero
        rsi_values = 100 - ((100 / (1 + self.df["RS"])))

        self.df = pd.DataFrame(None)

        return rsi_values


class SMA:
    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        info = ("Simple Moving Average is an arithmetic moving average which is"
                " calculated by taking the sum of values from recent time periods and then divided "
                "by number of time periods.")
        return info

    def get_value_df(self, df, close_col=None, time_period=21):
        if close_col is not None:
            if close_col < len(df.columns):
                df["SMA"] = df.iloc[:, close_col].rolling(
                    window=time_period).mean()
            else:
                # throw exception
                pass
        else:
            if "close" in df.columns:
                df["SMA"] = df["close"].rolling(window=time_period).mean()
            elif "Close" in df.columns:
                df["SMA"] = df["Close"].rolling(window=time_period).mean()
            elif "CLOSE" in df.columns:
                df["SMA"] = df["CLOSE"].rolling(window=time_period).mean()
            else:
                # throw exception
                pass

    def get_value_list(self, close_values, time_period=21):
        self.df["close"] = close_values
        sma_values = self.df["close"].rolling(window=time_period).mean()
        self.df = pd.DataFrame(None)
        return sma_values


class VLT:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Standard Deviation, variance and volatility are used to evaluate the volatility in the movement of the stock")
        return info

    def get_value_df(self, df, time_period=21, need_variance=True, need_deviation=True):
        df["SMA"] = df["close"].rolling(window=time_period).mean()
        df["SV"] = (df["close"] - df["SMA"]) ** 2
        df["SV"] = df["SV"].rolling(window=time_period).mean()

        df["SD"] = np.sqrt(df["SV"])

        df["VLT"] = df["SD"] / df["SV"]

        drop_columns = ["SMA"]
        if not need_variance:
            drop_columns.append(["SV"])
        if not need_deviation:
            drop_columns.append("SD")
        df = df.drop(drop_columns, axis=1)
        return df

    def get_value_list(self, close_values, time_period=21, need_variance=True, need_deviation=True):
        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["SMA"] = self.df["close"].rolling(window=time_period).mean()
        self.df["SV"] = (self.df["close"] - self.df["SMA"]) ** 2
        self.df["SV"] = self.df["SV"].rolling(window=time_period).mean()

        self.df["SD"] = np.sqrt(df["SV"])

        self.df["VLT"] = df["SD"] / df["SV"]

        standard_variance, standard_deviation, volatility = self.df[
            "SV"], self.df["SD"], self.df["VLT"]
        self.df = pd.DataFrame(None)

        return_values = [standard_variance]

        if need_variance:
            return_values.append(standard_deviation)
        if need_deviation:
            return_values.append(volatility)
        return tuple(return_values)


class StochasticKAndD:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Stochastic Oscillator is a momentum indicator comparing a particular price to a range of "
                "prices over specific period of time.")
        return info

    def get_value_df(self, df, time_period=14):
        df["highest high"] = df["high"].rolling(
            window=time_period).max()
        df["lowest low"] = df["low"].rolling(
            window=time_period).min()
        df["stoc_k"] = 100 * ((df["close"] - df["lowest low"]) /
                              (df["highest high"] - df["lowest low"]))
        df["stoc_d"] = df["stoc_k"].rolling(window=3).mean()

        df = df.drop(["highest high", "lowest low"], axis=1)
        return df

    def get_value_list(self, high_values, low_values, close_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df["highest high"] = self.df["high"].rolling(
            window=time_period).max()
        self.df["lowest low"] = self.df["low"].rolling(
            window=time_period).min()
        self.df["stoc_k"] = 100 * ((self.df["close"] - self.df["lowest low"]) /
                                   (self.df["highest high"] - self.df["lowest low"]))
        self.df["stoc_d"] = self.df["stoc_k"].rolling(window=3).mean()

        stochastic_k_values, stochastic_d_values = self.df["stoc_k"], self.df["stoc_d"]
        return stochastic_k_values, stochastic_d_values


class Trix:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Trix is triple exponential moving average, can be used as both oscillator and momentum indicator")
        return info

    def get_value_df(self, df, time_period=14):
        df["EMA1"] = df["close"].ewm(span=time_period).mean()
        df["EMA2"] = df["EMA1"].ewm(span=time_period).mean()
        df["EMA3"] = df["EMA2"].ewm(span=time_period).mean()
        df["EMA_prev"] = df["EMA3"].shift(1)

        df["TRIX"] = (df["EMA3"] - df["EMA_prev"]) / \
            df["EMA_prev"] * 100
        df = df.drop(["EMA1", "EMA2", "EMA3", "EMA_prev"], axis=1)
        return df

    def get_value_list(self, close_values, time_period=14):
        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["EMA1"] = self.df["close"].ewm(span=time_period).mean()
        self.df["EMA2"] = self.df["EMA1"].ewm(span=time_period).mean()
        self.df["EMA3"] = self.df["EMA2"].ewm(span=time_period).mean()
        self.df["EMA_prev"] = self.df["EMA3"].shift(1)

        trix_values = (self.df["EMA3"] - self.df["EMA_prev"]) / \
            self.df["EMA_prev"] * 100
        self.df = pd.DataFrame(None)

        return trix_values


class TR:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "True range is an essential component of determination of average true range")
        return info

    def get_value_df(self, df):
        df["prev_close"] = df["close"].shift(1)
        df["H-L"] = abs(df["high"] - df["low"])
        df["H-CP"] = abs(df["high"] - df["prev_close"])
        df["L-CP"] = abs(df["low"] - df["prev_close"])
        df['TR'] = df[["H-L", "H-CP", "L-CP"]].max(axis=1)

    def get_value_list(self, high_values, low_values, close_values):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df["prev_close"] = self.df["close"].shift(1)

        self.df["H-L"] = abs(self.df["high"] - self.df["low"])
        self.df["H-CP"] = abs(self.df["high"] - self.df["prev_close"])
        self.df["L-CP"] = abs(self.df["low"] - self.df["prev_close"])
        tr_values = self.df[["H-L", "H-CP", "L-CP"]].max(axis=1)
        self.df = pd.DataFrame(None)
        return tr_values


class TYP:
    def __init__(self):
        self.df = None

    def info(self):
        info = (
            "Typical Price is an average of low, high and close. It is used as an alternative to close price")
        return info

    def get_value_df(self, df):
        df["TYP"] = (df["high"] + df["low"] + df["close"]) / 3

        return df

    def get_value_list(self, high_values, low_values, close_values):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        typ_values = (self.df["high"] + self.df["low"] + self.df["close"]) / 3

        self.df = pd.DataFrame(None)
        return typ_values


class VHF:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df, time_period=28):
        df["PC"] = df["close"].shift(1)
        df["DIF"] = abs(df["close"] - df["PC"])

        df["HC"] = df["close"].rolling(window=time_period).max()
        df["LC"] = df["close"].rolling(window=time_period).min()

        df["HC-LC"] = abs(df["HC"] - df["LC"])

        df["DIF"] = df["DIF"].rolling(window=time_period).sum()

        df["VHF"] = df["HC-LC"] / df["DIF"]

        df = df.drop(["PC", "DIF", "HC", "LC", "HC-LC"], axis=1)

        return df

    def get_value_list(self, close_values, time_period=28):
        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["PC"] = self.df["close"].shift(1)
        self.df["DIF"] = abs(self.df["close"] - self.df["PC"])

        self.df["HC"] = self.df["close"].rolling(window=time_period).max()
        self.df["LC"] = self.df["close"].rolling(window=time_period).min()

        self.df["HC-LC"] = abs(self.df["HC"] - self.df["LC"])

        self.df["DIF"] = self.df["DIF"].rolling(window=time_period).sum()

        vhf_values = self.df["HC-LC"] / self.df["DIF"]
        self.df = pd.DataFrame(None)
        return vhf_values


class VO:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df, short_ema=9, long_ema=26):
        df["short_ema"] = df["volume"].ewm(span=short_ema).mean()
        df["long_ema"] = df["volume"].ewm(span=long_ema).mean()

        df["VO"] = ((df["short_ema"] - df["long_ema"]) /
                    df["long_ema"]) * 100
        df = df.drop(["short_ema", "long_ema"], axis=1)

        return df

    def get_value_list(self, volume_values, short_ema=9, long_ema=26):
        self.df = pd.DataFrame({
            "volume": volume_values
        })
        self.df["short_ema"] = self.df["volume"].ewm(span=short_ema).mean()
        self.df["long_ema"] = self.df["volume"].ewm(span=long_ema).mean()

        vo_values = ((self.df["short_ema"] - self.df["long_ema"]) /
                     self.df["long_ema"]) * 100
        self.df = pd.DataFrame(None)
        return vo_values


class ROCV:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df, time_period=12):
        df["prev_volume"] = df["volume"].shift(time_period)
        df["ROCV"] = (df["volume"] - df["prev_volume"]
                      ) / df["prev_volume"] * 100
        df = df.drop(["prev_volume"], axis=1)

        return df

    def get_value_list(self, volume_values, time_period=12):
        self.df = pd.DataFrame({
            "volume": volume_values
        })
        self.df["prev_volume"] = self.df["volume"].shift(time_period)
        rocv_values = (self.df["volume"] - self.df["prev_volume"]
                       ) / self.df["prev_volume"] * 100
        self.df = pd.DataFrame(None)

        return rocv_values


class WCL:
    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df):
        df["WCL"] = (df["high"] + df["low"] + (2 * df["close"])) / 4

    def get_value_list(self, high_values, low_values, close_values):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        wcl_values = (self.df["high"] + self.df["low"] +
                      (2 * self.df["close"])) / 4

        return wcl_values


class WilliamsR:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Williams R is tries to determine overbought and oversold levels of an asset")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["highest high"] = df["high"].rolling(
            window=time_period).max()
        self.df["lowest low"] = df["low"].rolling(
            window=time_period).min()
        df["Williams%R"] = 100 * (df["close"] - self.df["highest high"]) / \
            (self.df["highest high"] - self.df["lowest low"])
        self.df = pd.DataFrame(None)

    def get_value_list(self, high_values, low_values, close_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df["highest high"] = self.df["high"].rolling(
            window=time_period).max()
        self.df["lowest low"] = self.df["low"].rolling(
            window=time_period).min()
        williams_r_values = 100 * (self.df["close"] - self.df["highest high"]) / \
            (self.df["highest high"] - self.df["lowest low"])
        self.df = pd.DataFrame(None)
        return williams_r_values
