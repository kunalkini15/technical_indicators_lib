import pandas as pd
import numpy as np


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


class AccumulationDistribution:

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("The name accumulation/distribution comes from the idea that during accumulation buyers are in control and the price will be bid up through the day,"
                " or will make a recovery if sold down, in either case more often finishing near the day's high than the low. The opposite applies during distribution. ")
        return info

    def get_value_df(self, df):
        self.df["CLV"] = ((df["close"] - df["low"]) - (df["high"] -
                                                       df["close"])) / (df["high"] - df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * df["volume"]
        ad_values = [self.df["CLV_VOL"][0]]
        for i in range(1, len(df)):
            ad_values.append(ad_values[i-1] + self.df["CLV_VOL"][i])
        df["AD"] = ad_values
        self.df = pd.DataFrame(None)

    def get_value_list(self, high_values, low_values, close_values, volume_values):
        # prev_ad_value = 0
        # ad_values = []
        # for i in range(len(close_values)):
        #     clv = ((close_values[i] - low_values[i]) - (high_values[i] -
        #                                                 close_values[i])) / (high_values[i] - low_values[i])
        #     current_ad_value = prev_ad_value + volume_values[i] * clv
        #     ad_values.append(current_ad_value)
        #     prev_ad_value = current_ad_value

        # return ad_values
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": volume_values
        })
        self.df["CLV"] = ((self.df["close"] - self.df["low"]) - (self.df["high"] -
                                                                 self.df["close"])) / (self.df["high"] - self.df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * self.df["volume"]
        ad_values = [self.df["CLV_VOL"][0]]
        for i in range(1, len(self.df)):
            ad_values.append(ad_values[i-1] + self.df["CLV_VOL"][i])
        self.df = pd.DataFrame(None)
        return ad_values

# ----------------------------------- Need to test ---------------------------------------

# class AverageTrueRange:
#     def __init__(self):
#         self.df = pd.DataFrame()

#     def info(self):
#         info = ("Average True Range is a volatility indicator which provides degree of price of volatility making use of smoothed moving average of true ranges.")
#         return info

#     def get_value_df(self, df, time_period=14):
#         avg_true_range = [np.nan for i in range(time_period)]

#         first_atr = 0
#         for i in range(time_period):
#             if i == 0:
#                 first_atr += abs(df.iloc[i]["high"] - df.iloc[i]["low"])
#                 continue
#             first_atr += max(df.iloc[i]["high"], df.iloc[i-1]["close"]) - \
#                 min(df.iloc[i]["low"], df.iloc[i-1]["close"])
#         avg_true_range.append(first_atr / time_period)

#         for i in range(time_period+1, len(df)):
#             high = df.iloc[i]["high"]
#             low = df.iloc[i]["low"]
#             prev_close = df.iloc[i-1]["close"]
#             current_tr = max(high, prev_close) - min(low, prev_close)
#             current_atr = (
#                 ((time_period-1) * avg_true_range[i-1]) + current_tr) / time_period
#             avg_true_range.append(current_atr)

#         df["ATR"] = avg_true_range

#     def get_value_list(self, high_values, low_values, close_values, time_period=14):
#         avg_true_range = [np.nan for i in range(time_period)]

#         first_atr = 0
#         for i in range(time_period):
#             if i == 0:
#                 first_atr += abs(high_values[i] - low_values[i])
#                 continue
#             first_atr += max(high_values[i], close_values[i-1]) - \
#                 min(low_values[i], close_values[i-1])
#         avg_true_range.append(first_atr / time_period)

#         for i in range(time_period+1, len(df)):
#             high = high_values[i]
#             low = low_values[i]
#             prev_close = close_values[i-1]
#             current_tr = max(high, prev_close) - min(low, prev_close)
#             current_atr = (
#                 ((time_period-1) * avg_true_range[i-1]) + current_tr) / time_period
#             avg_true_range.append(current_atr)

#         df["ATR"] = avg_true_range

class AverageTrueRange:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Average True Range is a volatility indicator which provides degree of price of volatility making use of smoothed moving average of true ranges.")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["high"] = df["high"]
        self.df["low"] = df["low"]
        self.df["close"] = df["close"]
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["TR"] = np.max(self.df["high"], self.df["close_prev"]) - np.min(self.df["low"] - self.df["close_prev"])

        df["ATR"] = self.df["TR"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)

    def get_value_list(self, high_values, low_values, close_values, time_period=14):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })

        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["TR"] = np.max(self.df["high"], self.df["close_prev"]) - np.min(self.df["low"] - self.df["close_prev"])

        atr_values = self.df["TR"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)

class ChaikinMoneyFlow:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Chaikin Money flow is used to measure money flow volume over a certain time periods.")
        return info

    def get_value_df(self, df, time_period=21):
        self.df["CLV"] = (2 * df["close"] - (df["high"] +
                                             df["low"])) / (df["high"] - df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * df["volume"]
        self.df["CLV_VOL_SUM"] = self.df["CLV_VOL"].rolling(
            window=time_period).sum()
        self.df["VOL_SUM"] = df["volume"].rolling(
            window=time_period).sum()

        df["CMF"] = self.df["CLV_VOL_SUM"] / self.df["VOL_SUM"]
        self.df = pd.DataFrame(None)

    def get_value_list(self, high_values, low_values,  close_values, volume_values, time_period=21):
        # clv_values = []
        # for i in range(len(df)):
        #     clv = (2 * close_values[i] - (high_values[i] +
        #                                   low_values[i])) / (high_values[i] - low_values[i])
        #     clv_vol = clv * volume_values[i]
        #     clv_values.append(clv_vol)
        # clv_sum = [np.nan for i in range(time_period)]
        # vol_sum = [np.nan for i in range(time_period)]
        # for i in range(time_period, len(df)):
        #     clv_sum.append(clv_values[i-time_period+1: i+1])
        #     vol_sum.append(volume_values[i-time_period+1: i+1])
        # cmf_values = []
        # for i in range(len(clv_sum)):
        #     cmf_values.append(clv_sum / vol_sum)
        # return cmf_values
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


class ChaikinOscillator:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Chaikin oscillator is designed to anticipate the directional changes in Accumulation "
                "distributin line by measuring the momentum behind the movements.")
        return info

    def get_value_df(self, df, short_time_period=3, long_time_period=10):
        self.df["AD"] = ((2 * df["close"] - (df["high"] + df["low"])
                          ) / (df["high"] - df["low"])) * df["volume"]
        self.df["AD_short"] = self.df["AD"].ewm(span=short_time_period).mean()
        self.df["AD_long"] = self.df["AD"].ewm(span=long_time_period).mean()

        df["CHO"] = self.df["AD_short"] - self.df["AD_long"]
        self.df = pd.DataFrame(None)

    # def get_value_list(self, high_values, low_values, close_values, short_time_period=3, long_time_period=10):
    #     self.df = pd.DataFrame({
    #         "close": close_values,
    #         "high": high_values,
    #         "low": low_values
    #     })

    #     self.df["AD"] = ((2 * self.df["close"] - (self.df["high"] + self.df["low"])
    #                       ) / (self.df["high"] - self.df["low"])) * self.df["volume"]
    #     self.df["AD_short"] = self.df["AD"].ewm(span=short_time_period).mean()
    #     self.df["AD_long"] = self.df["AD"].ewm(span=long_time_period).mean()
    #     self.df["CHO"] = self.df["AD_short"] - self.df["AD_long"]

    #     cho_values = self.df["CHO"]
    #     self.df = None

    #     return cho_values

    def get_value_list(self, high_values, low_values, close_values, short_time_period=3, long_time_period=10):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df["AD"] = ((2 * self.df["close"] - (self.df["high"] + self.df["low"])
                          ) / (self.df["high"] - self.df["low"])) * self.df["volume"]
        self.df["AD_short"] = self.df["AD"].ewm(span=short_time_period).mean()
        self.df["AD_long"] = self.df["AD"].ewm(span=long_time_period).mean()

        cho_values = self.df["AD_short"] - self.df["AD_long"]

        self.df =  pd.DataFrame(None)
        return cho_values


class ChaikinVolatility:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Chaikin Volatility determines the volatility of instrument using percentage change in a moving average of difference "
                "between high price and the low price over a specific period of time.")
        return info

    def get_value_df(self, df, time_period=10):
        self.df["difference"] = df["high"] - df["low"]
        self.df["difference_EMA"] = self.df["difference"].ewm(
            span=time_period).mean()
        self.df["difference_EMA_n_periods_ago"] = self.df["difference_EMA"].shift(
            time_period)
        df["CHV"] = (self.df["difference_EMA"] - self.df["difference_EMA_n_periods_ago"]
                     ) / self.df["difference_EMA_n_periods_ago"] * 100
        self.df = pd.DataFrame(None)

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
        self.df["CHV"] = (self.df["difference_EMA"] - self.df["difference_EMA_n_periods_ago"]
                          ) / self.df["difference_EMA_n_periods_ago"] * 100
        chv_values = self.df["CHV"]
        self.df = pd.DataFrame(None)
        return chv_values


class DetrendPriceOscillator:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Chaikin Volatility determines the volatility of instrument using percentage change in a moving average of difference "
                "between high price and the low price over a specific period of time.")
        return info

    def get_value_df(self, df, time_period=21):
        self.df["DPO_SMA"] = df["close"].rolling(
            window=int(time_period/2 + 1)).mean()
        df["DPO"] = df["close"] - self.df["DPO_SMA"]
        self.df = pd.DataFrame(None)

    def get_value_list(self, close_values, time_period=21):
        self.df["close"] = close_values
        self.df["DPO_SMA"] = df["close"].rolling(
            window=int(time_period/2 + 1)).mean()
        self.df["DPO"] = self.df["close"] - self.df["DPO_SMA"]

        dpo_values = self.df["DPO"]
        self.df = pd.DataFrame(None)
        return dpo_values


class EaseOfMovement:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Ease of movement tries to identify amount of volume needed to move prices.")
        return info

    def get_value_df(self, df, volume_divisor=1000000):
        self.df["H+L"] = df["high"] + df["low"]
        self.df["H+L_prev"] = self.df["H+L"].shift(1)
        self.df["MIDPT"] = (self.df["H+L"] / 2 - self.df["H+L_prev"] / 2)
        self.df["BOXRATIO"] = (
            (df["volume"] / volume_divisor) / (df["high"] - df["low"]))

        df["EMV"] = self.df["MIDPT"] / self.df["BOXRATIO"]
        self.df = pd.DataFrame(None)

    # def get_value_list(self, high_values, low_values, volume_values, volume_divisor=1000000):
    #     emv_values = [np.nan]
    #     for i in range(1, len(df)):
    #         mid_pt_move = ((high_values[i] + low_values[i])/2) - \
    #             ((high_values[i-1] + low_values[i-1]) / 2)
    #         box_ratio = (volume_values[i] / volume_divisor) / \
    #             (high_values[i] - low_values[i])
    #         emv_values.append(mid_pt_move / box_ratio)

    #     return emv_values
    def get_value_list(self, high_values, low_values, volume_values, volume_divisor=1000000):
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

        emv_values = self.df["MIDPT"] / self.df["BOXRATIO"]
        self.df = pd.DataFrame(None)
        return emv_values


class ForceIndex:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Force index tries to determine the amount of power used to move the price of an asset")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["close_prev"] = df["close"].shift(1)
        df["fi"] = (df["close"] -
                    self.df["close_prev"]) * df["volume"]
        df["fi"] = df["fi"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)


    def get_value_dist(self, close_values, volume_values, time_period=14):
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["fi"] = (self.df["close"] -
                         self.df["close_prev"]) * self.df["volume"]
        self.df["fi"] = self.df["fi"].ewm(span=time_period).mean()

        fi_values = self.df["fi"]
        self.df = pd.DataFrame(None)
        return fi_values


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


    # def get_value_list(self, close_values, high_values, low_values, time_period=14):
    #     wil_values = [np.nan for i in range(time_period)]
    #     for i in range(time_period, len(close_values)):
    #         highest_high = np.max(high_values[i-time_period+1: i+1])
    #         lowest_low = np.min(low_values[i-time_period+1: i+1])
    #         current_r_value = 100 * \
    #             (close_values[i] - highest_high) / (highest_high - lowest_low)
    #         wil_values.append(current_r_value)
    #     return wil_values
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


class MassIndex:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Mass index tries to determine the range of high and low values over a specified period of time")
        return info

    def get_value_df(self, df, time_period=25, ema_time_period=9):
        self.df["difference"] = df["high"] - df["low"]
        self.df["difference_EMA"] = self.df["difference"].ewm(
            span=ema_time_period).mean()
        self.df["difference_double_EMA"] = self.df["difference_EMA"].ewm(
            span=ema_time_period).mean()
        df["MI"] = self.df["difference_EMA"] / \
            self.df["difference_double_EMA"]
        df["MI"] = df["MI"].rolling(window=time_period).sum()
        self.df = pd.DataFrame(None)


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


class MedianPrice:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Median determines the mid point of the price range")
        return info

    def get_value_df(self, df):
        df["MED"] = (df["high"] + df["low"]) / 2

    def get_value_list(self, high_values, low_values):
        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values
        })
        self.df["MED"] = (self.df["high"] + self.df["low"]) / 2
        med_values = self.df["MED"]
        self.df = pd.DataFrame(None)
        return med_values


class Momentum:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Momentum helps to determine th price changes from one period to another.")
        return info

    def get_value_df(self, df, time_period=1):
        self.df["close_prev"] = df["close"].shift(time_period)
        df["MOM"] = df["close"] - self.df["close_prev"]
        self.df = pd.DataFrame(None)


    def get_value_list(self, close_values, time_period=1):
        self.df["close"] = close_values
        self.df["prev_close"] = self.df["close"].shift(time_period)
        self.df["MOM"] = self.df["close"] - self.df["close_prev"]
        mom_values = self.df["MOM"]
        self.df = pd.DataFrame(None)
        return mom_values


class MoneyFlowIndex:
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
        mfi_values = [np.nan for i in range(time_period)]
        self.df["RMF"] = self.df["TP"] * df["volume"]
        for i in range(time_period, len(self.df)):
            pmf, nmf = 0, 0
            for j in range(i-time_period+1, i+1):
                if self.df["RMF"][j] is np.nan:
                    continue
                if self.df["PORN"][j] == 0.0:
                    nmf += self.df["RMF"][j]
                else:
                    pmf += self.df["RMF"][j]
            mfratio = pmf / (nmf+0.0000001)

            mfi_values.append(100 - (100 / (1 + mfratio)))
        df["MFI"] = mfi_values

    def get_value_df(self, high_values, low_values, close_values, volume_values, time_period=14):
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
        mfi_values = [np.nan for i in range(time_period)]
        self.df["RMF"] = self.df["TP"] * self.df["volume"]
        for i in range(time_period, len(self.df)):
            pmf, nmf = 0, 0
            for j in range(i-time_period+1, i+1):
                if self.df["RMF"][j] is np.nan:
                    continue
                if self.df["PORN"][j] == 0.0:
                    nmf += self.df["RMF"][j]
                else:
                    pmf += self.df["RMF"][j]
            mfratio = pmf / (nmf+0.0000001)

            mfi_values.append(100 - (100 / (1 + mfratio)))
        self.df = None
        return mfi_values


class MovingAverageConvergenceDivergence:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Moving Average Convergence is a trend following momentum indicator that "
                "shows a relationship between two moving averages of an asset")
        return info

    def get_value_df(self, df):
        self.df["26EWMA"] = df["close"].ewm(span=26).mean()
        self.df["12EWMA"] = df["close"].ewm(span=12).mean()

        df["MACD"] = self.df["26EWMA"] - self.df["12EWMA"]
        df["MACD_signal_line"] = df["MACD"].ewm(span=9).mean()


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


class NegativeVolumeIndex:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Negative Volume Index helps in identifying trends and reversals.")
        return info

    def get_value_df(self, df,  starting_value=100):
        nvi_values = [starting_value]

        for i in range(1, len(df)):
            if df.iloc[i]["volume"] >= df.iloc[i-1]["volume"]:
                nvi_values.append(nvi_values[i-1])
            else:
                nvi_values.append(
                    nvi_values[i-1] * (1 + ((df.iloc[i]["close"] - df.iloc[i-1]["close"]) / df.iloc[i-1]["close"])))

        df["NVI"] = nvi_values


class OnBalanceVolume:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("On Balance Volume provides the signal whether the volume is flowing in or out of a given security.")
        return info

        def on_balance_volume(self, df):
            obv_values = [df.iloc[0]["volume"]]
            for i in range(1, len(df)):
                if df.iloc[i]["close"] > df.iloc[i-1]["close"]:
                    obv_values.append(obv_values[i-1] + df.iloc[i]["volume"])
                elif df.iloc[i]["close"] < df.iloc[i-1]["close"]:
                    obv_values.append(obv_values[i-1] - df.iloc[i]["volume"])
                else:
                    obv_values.append(obv_values[i-1])

            df["OBV"] = obv_values


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

        df["DI-"] = self.df["DM+smoothd"] / self.df["ATR"]


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


class PriceVolumeTrend:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Price Volume Trend helps in identifying trend by using cumulative volume adjusted by change in price")
        return info

    def get_value_df(self, df):
        pvt_values = [df.iloc[0]["volume"]]
        for i in range(1, len(df)):
            pvt_values.append((((df.iloc[i]["close"] - df.iloc[i-1]["close"]) /
                                df.iloc[i-1]["close"]) * df.iloc[i]["volume"]) + pvt_values[i-1])

        df["PVT"] = pvt_values


class PriceChannels:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Price channels forms a boundary and between them the close price of an asset is oscillating")
        return info

    def get_value_df(self, df, percent_value=6, ema_period=21):
        self.df["EMA_FOR_PC"] = df["close"].ewm(span=ema_period).mean()

        df["PC_upper"] = self.df["EMA_FOR_PC"] * (1 + (percent_value / 100))
        df["PC_lower"] = self.df["EMA_FOR_PC"] * (1 - (percent_value / 100))


class PriceOscillator:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Price oscillator is a momentum osciallator which shows a difference between two moving averages")
        return info

    def get_value_df(self, df, short_ema_period=9, long_ema_period=26):
        self.df["Short_EMA"] = df["close"].ewm(
            span=short_ema_period).mean()
        self.df["Long_EMA"] = df["close"].ewm(span=long_ema_period).mean()

        df["PO"] = ((self.df["Short_EMA"] - self.df["Long_EMA"]) /
                    self.df["Long_EMA"]) * 100


class RateOfChange:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Rate of change helps in calculation of speed of ascent or descent.")
        return info

    def get_value_df(self, df, time_period=7):
        self.df["close_prev"] = df["close"].shift(time_period)

        df["ROC"] = (df["close"] - self.df["close_prev"]) / \
            self.df["close_prev"]


class RelativeStrengthIndex:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Relative Strength Index is used to generate oversold and overbought signals.")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["close_prev"] = df["close"].shift(1)
        self.df["GAIN"] = np.zeros(len(self.df))
        self.df["LOSS"] = np.zeros(len(self.df))

        self.df.loc[df["close"] > self.df["close_prev"],
                    "GAIN"] = df["close"] - self.df["close_prev"]
        self.df.loc[self.df["close_prev"] > df["close"],
                    "LOSS"] = self.df["close_prev"] - df["close"]
        self.df["AVG_GAIN"] = self.df["GAIN"].ewm(span=time_period).mean()
        self.df["AVG_LOSS"] = self.df["LOSS"].ewm(span=time_period).mean()

        self.df["RS"] = self.df["AVG_GAIN"] / \
            (self.df["AVG_LOSS"] + 0.000001)  # to avoid divide by zero

        df["RSI"] = 100 - ((100 / (1 + self.df["RS"])))


class StandardDeviationVarianceAndVolatility:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Standard Deviation, variance and volatility are used to evaluate the volatility in the movement of the stock")
        return info

    def get_value_df(self, df, time_period=21):
        self.df["SMA"] = df["close"].rolling(window=time_period).mean()
        df["SV"] = (df["close"] - self.df["SMA"]) ** 2
        df["SV"] = df["SV"].rolling(window=time_period).mean()

        df["SD"] = np.sqrt(df["SV"])

        df["VLT"] = df["SD"] / df["SV"]


class StochasticKAndD:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("Stochastic Oscillator is a momentum indicator comparing a particular price to a range of "
                "prices over specific period of time.")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["highest high"] = df["high"].rolling(
            window=time_period).max()
        self.df["lowest low"] = df["low"].rolling(
            window=time_period).min()
        df["stoc_k"] = 100 * ((df["close"] - self.df["lowest low"]) /
                              (self.df["highest high"] - self.df["lowest low"]))
        df["stoc_d"] = df["stoc_k"].rolling(window=3).mean()


class Trix:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "Trix is triple exponential moving average, can be used as both oscillator and momentum indicator")
        return info

    def get_value_df(self, df, time_period=14):
        self.df["EMA1"] = df["close"].ewm(span=time_period).mean()
        self.df["EMA2"] = self.df["EMA1"].ewm(span=time_period).mean()
        self.df["EMA3"] = self.df["EMA2"].ewm(span=time_period).mean()
        self.df["EMA_prev"] = self.df["EMA3"].shift(1)

        df["trix"] = (self.df["EMA3"] - self.df["EMA_prev"]) / \
            self.df["EMA_prev"]


class TrueRange:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = (
            "True range is an essential component of determination of average true range")
        return info

    def get_value_df(self, df):
        self.df["prev_close"] = self.df["close"].shift(1)

        df["TR"] = max(
            abs(df["high"] - df["low"]),
            abs(df["high"] - self.df["prev_close"]),
            abs(df["low"] - self.df["prev_close"])
        )


class TypicalPrice:
    def __init__(self):
        self.df = None

    def info(self):
        info = (
            "Typical Price is an average of low, high and close. It is used as an alternative to close price")
        return info

    def get_value_df(self, df):
        df["TYP"] = (df["high"] + df["low"] + df["close"]) / 3


class vertical_horizontal_filter:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df, time_period=28):
        self.df["PC"] = df["close"].shift(1)
        self.df["DIF"] = df["close"] - self.df["PC"]

        self.df["HC"] = df["close"].rolling(window=time_period).max()
        self.df["LC"] = df["close"].rolling(window=time_period).min()

        self.df["HC-LC"] = abs(self.df["HC"] - self.df["LC"])

        self.df["DIF"] = self.df["DIF"].rolling(window=time_period).sum()

        df["VHF"] = self.df["HC-LC"] / self.df["DIF"]


class VolumeOscillator:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df, short_ema=9, long_ema=26):
        self.df["short_ema"] = df["volume"].ewm(span=short_ema).mean()
        self.df["long_ema"] = df["volume"].ewm(span=long_ema).mean()

        df["VO"] = ((self.df["short_ema"] - self.df["long_ema"]) /
                    self.df["long_ema"]) * 100


class VolumeRateOfChange:
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df, time_period=12):
        self.df["prev_volume"] = df["volume"].shift(time_period)
        df["ROCV"] = (df["volume"] - df["prev_volume"]
                      ) / df["prev_volume"] * 100


class WeightedClose:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info

    def get_value_df(self, df):
        df["WCL"] = (df["high"] + df["low"] + (2 * df["close"])) / 4
