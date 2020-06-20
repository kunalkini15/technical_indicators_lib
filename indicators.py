import pandas as pd
import numpy as np


class SMA:
    def __init__(self):
        pass

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
        sma_values = [np.nan for i in range(time_period)]
        for i in range(time_period, len(close_values)):
            sma_values.append(
                np.sum(close_values[i-time_period+1:i+1]) / time_period)
        return sma_values


class EMA:
    def __init__(self):
        pass

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


class AccumulationDistribution:

    def __init__(self):
        self.df = None

    def info(self):
        info = ("The name accumulation/distribution comes from the idea that during accumulation buyers are in control and the price will be bid up through the day,"
                " or will make a recovery if sold down, in either case more often finishing near the day's high than the low. The opposite applies during distribution. ")
        return info

    def get_value_df(self, df):
        self.df = df.copy()
        self.df["CLV"] = ((self.df["close"] - self.df["low"]) - (self.df["high"] -
                                                                 self.df["close"])) / (self.df["high"] - self.df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * self.df["volume"]
        ad_values = [self.df["CLV_VOL"][0]]
        for i in range(1, len(df)):
            ad_values.append(ad_values[i-1] + self.df["CLV_VOL"][i])
        df["AD"] = ad_values

    def get_value_list(self, close_values, high_values, low_values, volume_values):
        prev_ad_value = 0
        ad_values = []
        for i in range(len(close_values)):
            clv = ((close_values[i] - low_values[i]) - (high_values[i] -
                                                        close_values[i])) / (high_values[i] - low_values[i])
            current_ad_value = prev_ad_value + volume_values[i] * clv
            ad_values.append(current_ad_value)
            prev_ad_value = current_ad_value

        return ad_values


class AverageTrueRange:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Average True Range is a volatility indicator which provides degree of price of volatility making use of smoothed moving average of true ranges.")
        return info

    def get_value_df(self, df, time_period=14):
        avg_true_range = [np.nan for i in range(time_period)]

        first_atr = 0
        for i in range(time_period):
            if i == 0:
                first_atr += abs(df.iloc[i]["high"] - df.iloc[i]["low"])
                continue
            first_atr += max(df.iloc[i]["high"], df.iloc[i-1]["close"]) - \
                min(df.iloc[i]["low"], df.iloc[i-1]["close"])
        avg_true_range.append(first_atr / time_period)

        for i in range(time_period+1, len(df)):
            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]
            prev_close = df.iloc[i-1]["close"]
            current_tr = max(high, prev_close) - min(low, prev_close)
            current_atr = (
                ((time_period-1) * avg_true_range[i-1]) + current_tr) / time_period
            avg_true_range.append(current_atr)

        df["ATR"] = avg_true_range


class ChaikinMoneyFlow:
    def __init__(self):
        self.df = None

    def info(self):
        info = (
            "Chaikin Money flow is used to measure money flow volume over a certain time periods.")
        return info

    def get_value_df(self, df, time_period=21):
        self.df = df.copy()
        self.df["CLV"] = (2 * self.df["close"] - (self.df["high"] +
                                                  self.df["low"])) / (self.df["high"] - self.df["low"])
        self.df["CLV_VOL"] = self.df["CLV"] * self.df["volume"]
        self.df["CLV_VOL_SUM"] = self.df["CLV_VOL"].rolling(
            window=time_period).sum()
        self.df["VOL_SUM"] = self.df["volume"].rolling(
            window=time_period).sum()

        df["CMF"] = self.df["CLV_VOL_SUM"] / self.df["VOL_SUM"]


class ChaikinOscillator:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Chaikin oscillator is designed to anticipate the directional changes in Accumulation "
                "distributin line by measuring the momentum behind the movements.")
        return info

    def get_value_df(self, df, short_time_period=3, long_time_period=10):
        self.df = df.copy()
        self.df["AD"] = ((2 * self.df["close"] - (self.df["high"] + self.df["low"])
                          ) / (self.df["high"] - self.df["low"])) * df["volume"]
        self.df["AD_short"] = self.df["AD"].ewm(span=short_time_period).mean()
        self.df["AD_long"] = self.df["AD"].ewm(span=long_time_period).mean()

        df["CHO"] = self.df["AD_short"] - self.df["AD_long"]


class ChaikinVolatility:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Chaikin Volatility determines the volatility of instrument using percentage change in a moving average of difference "
                "between high price and the low price over a specific period of time.")
        return info

    def get_value_df(self, df, n=10):
        self.df = df.copy()
        self.df["difference"] = self.df["high"] - self.df["low"]
        self.df["difference_EMA"] = self.df["difference"].ewm(span=n).mean()
        self.df["difference_EMA_n_periods_ago"] = self.df["difference_EMA"].shift(
            n)
        df["CHV"] = (self.df["difference_EMA"] - self.df["difference_EMA_n_periods_ago"]
                     ) / self.df["difference_EMA_n_periods_ago"] * 100

class DetrendPriceOscillator:
  def __init__(self):
    self.df = None

  def info(self):
    info = ("Chaikin Volatility determines the volatility of instrument using percentage change in a moving average of difference " 
              "between high price and the low price over a specific period of time.")
    return info
    
  def get_value_df(self, df, n=21):
    self.df["DPO_SMA"] = self.df["close"].rolling(window=int(n/2 +1)).mean()
    df["DPO"] = self.df["close"] - self.df["DPO_SMA"]


class EaseOfMovement:
  def __init__(self):
    self.df = None

  def info(self):
    info = ("Ease of movement tries to identify amount of volume needed to move prices.")
    return info
    

  def get_value_df(self, df, volume_divisor=1000000):
    self.df = df.copy()
    self.df["H+L"] = self.df["high"] + self.df["low"]
    self.df["H+L_prev"] = self.df["H+L"].shift(1)
    self.df["MIDPT"] = (self.df["H+L"] / 2 - self.df["H+L_prev"] / 2)
    self.df["BOXRATIO"] = ((self.df["volume"] / volume_divisor)/ (self.df["high"] - self.df["low"]))

    df["EMV"] = self.df["MIDPT"] / self.df["BOXRATIO"]



  def get_value_list(self,high_values, low_values, volume_values, volume_divisor=1000000):
    emv_values = [np.nan]
    for i in range(1, len(df)):
      mid_pt_move = ((high_values[i] + low_values[i])/2) - ((high_values[i-1] + low_values[i-1]) / 2)
      box_ratio = (volume_values[i] / volume_divisor) / (high_values[i] - low_values[i])
      emv_values.append(mid_pt_move / box_ratio)

    return emv_values


class ForceIndex:
  def __init__(self):
    self.df = None

  def info(self):
    info = ("Force index tries to determine the amount of power used to move the price of an asset")
    return info
    

  def get_value_df(self, df, time_period=14):
    self.df = df.copy()
    self.df["close_prev"] = self.df["close"].shift(1)
    self.df["fi"] = (self.df["close"] - self.df["close_prev"]) * self.df["volume"]
    df["fi"] = self.df["FI"].ewm(span=time_period).mean()

class WilliamsR:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Williams R is tries to determine overbought and oversold levels of an asset")
        return info
        

    def get_value_df(self, df, time_period=14):
        self.df = df.copy()
        self.df["highest high"] = self.df["high"].rolling(window=time_period).max()
        self.df["lowest low"] = self.df["low"].rolling(window=time_period).min()
        df["Williams%R"] = 100 * (self.df["close"] - self.df["highest high"]) / (self.df["highest high"] - self.df["lowest low"])

    def get_value_list(self, close_values, high_values, low_values, time_period=14):
        wil_values=[np.nan for i in range(time_period)]
        for i in range(time_period, len(close_values)):
            highest_high = np.max(high_values[i-time_period+1: i+1])
            lowest_low = np.min(low_values[i-time_period+1 : i+1])
            current_r_value = 100 * (close_values[i] - highest_high) / (highest_high - lowest_low)
            wil_values.append(current_r_value)
        return wil_values

class MassIndex:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Mass index tries to determine the range of high and low values over a specified period of time")
        return info

    def get_value_df(self, df, time_period=25, ema_time_period=9):
        self.df = df.copy()
        self.df["difference"] = self.df["high"] - self.df["low"]
        self.df["difference_EMA"] = self.df["difference"].ewm(span=ema_time_period).mean()
        self.df["difference_double_EMA"] = self.df["difference_EMA"].ewm(span=ema_time_period).mean()
        self.df["MI"] = self.df["difference_EMA"] / self.df["difference_double_EMA"]
        df["MI"] = self.df["MI"].rolling(window=time_period).sum()

class MedianPrice:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Median determines the mid point of the price range")
        return info

    def get_value_df(self, df):
        df["MED"] = (df["high"] + df["low"]) / 2

class Momentum:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("Momentum helps to determine th price changes from one period to another.")
        return info

    def get_value_df(self, df, time_period=1):
        self.df = df.copy()
        self.df["close_prev"] = self.df["close"].shift(time_period)
        df["MOM"] = df["close"] - df["close_prev"]



class MoneyFlowIndex:
  def __init__(self):
    self.df = None

  def info(self):
    info = ("Money flow index uses price and volume data to for identifying overbought and oversold signals of an asset")
    return info
    

  def get_value_df(self, df, time_period=14):
    self.df = df.copy()
    self.df["TP"] = (self.df["low"] + self.df["high"] + self.df["close"] )/ 3
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
    df["MFI"] = mfi_values



class MovingAverageConvergenceDivergence:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
        

    def get_value_df(self, df):
        self.df = df.copy()
        self.df["26EWMA"] = self.df["close"].ewm(span=26).mean()
        self.df["12EWMA"] = self.df["close"].ewm(span=12).mean()

        df["MACD"] = self.df["26EWMA"] - self.df["12EWMA"]
        df["MACD_signal_line"] = df["MACD"].ewm(span=9).mean()


class NegativeDirectionIndicator:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
    

    def get_value_df(self, df, time_period=14):
        self.df = df.copy()
        self.df["DM-"] = np.zeros(len(df))
        self.df["low_prev"] = self.df["low"].shift(1)
        self.df["high_prev"] = self.df["high"].shift(1)


        self.df.loc[(self.df["low_prev"]-self.df["low"]) > (self.df["high"] - self.df["high_prev"]), "DM-"] = self.df["low_prev"] - self.df["low"]

        self.df["DM-smoothed"] = self.df["DM-"].rolling(window=time_period).sum()

        if "ATR" not in self.df.columns:
            AverageTrueRange().get_value_df(self.df)

        df["DI-"] = self.df["DM-smoothd"] / self.df["ATR"]

class NegativeVolumeIndex:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
    

    def get_value_df(self, df,  starting_value=100):
        nvi_values = [starting_value]

        for i in range(1, len(df)):
            if df.iloc[i]["volume"] >= df.iloc[i-1]["volume"]:
                nvi_values.append(nvi_values[i-1])
            else:
                nvi_values.append(nvi_values[i-1] * ( 1+ ((df.iloc[i]["close"] - df.iloc[i-1]["close"]) / df.iloc[i-1]["close"] )))
        
        df["NVI"] = nvi_values

class OnBalanceVolume:
  def __init__(self):
    self.df = None

  def info(self):
    info = ("")
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
        self.df = None

    def info(self):
        info = ("")
        return info
    

    def get_value_df(self, df, time_period=14):
        self.df = df.copy()
        self.df["DM+"] = np.zeros(len(df))
        self.df["low_prev"] = self.df["low"].shift(1)
        self.df["high_prev"] = self.df["high"].shift(1)


        self.df.loc[(self.df["low_prev"]-self.df["low"]) < (self.df["high"] - self.df["high_prev"]), "DM-"] = self.df["high"] - self.df["high_prev"]

        self.df["DM+smoothed"] = self.df["DM+"].rolling(window=time_period).sum()

        if "ATR" not in self.df.columns:
            AverageTrueRange().get_value_df(self.df)

        df["DI-"] = self.df["DM+smoothd"] / self.df["ATR"]
    
class PositiveVolumeIndex:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
    

    def get_value_df(self, df, starting_value=100):
        pvi_values = [starting_value]

        for i in range(1, len(df)):
            if df.iloc[i]["volume"] <= df.iloc[i-1]["volume"]:
                pvi_values.append(pvi_values[i-1])
            else:
                pvi_values.append( (1 + ((df.iloc[i]["close"] -df.iloc[i-1]["close"]) / df.iloc[i-1]["close"])) * pvi_values[i-1])
        
        df["PVI"] = pvi_values

class PriceVolumeTrend:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
    
    def get_value_df(self, df):
        pvt_values = [df.iloc[0]["volume"]]
        for i in range(1, len(df)):
            pvt_values.append((((df.iloc[i]["close"] - df.iloc[i-1]["close"]) / df.iloc[i-1]["close"]) * df.iloc[i]["volume"]) + pvt_values[i-1])

        df["PVT"] = pvt_values

class PriceChannels:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
    
    def get_value_df(self, df, percent_value=6, ema_period=21):
        self.df = df.copy()
        self.df["EMA_FOR_PC"] = df["close"].ewm(span=ema_period).mean()

        df["PC_upper"] = self.df["EMA_FOR_PC"] * (1 + (percent_value / 100))
        df["PC_lower"] = self.df["EMA_FOR_PC"] * (1 - (percent_value / 100))

class PriceOscillator:
    def __init__(self):
        self.df = None

    def info(self):
        info = ("")
        return info
    
    def price_oscillator(self, df, short_ema_period=9, long_ema_period=26):
        self.df = df.copy()
        self.df["Short_EMA"] = self.df["close"].ewm(span=short_ema_period).mean()
        self.df["Long_EMA"] = self.df["close"].ewm(span=long_ema_period).mean()

        df["PO"] = ((self.df["Short_EMA"] - self.df["Long_EMA"]) / self.df["Long_EMA"]) * 100

