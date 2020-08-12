"""
.. module:: indicators
    :synopsis: Technical Indicators

.. moduleauthor:: Kunal Kini

"""

import pandas as pd
import numpy as np


class ADI:
    """
    ADI -> Accumulation Distribution Index

    The name accumulation/distribution comes from the idea that during accumulation buyers are in control and the price will be bid up through the day,
    or will make a recovery if sold down, in either case more often finishing near the day's high than the low. The opposite applies during distribution.
    \nLinks:
    http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/AccumulationDistribution.php5\n
    https://www.investopedia.com/terms/a/accumulationdistribution.asp\n
    https://en.wikipedia.org/wiki/Accumulation/distribution_index
    """

    def __init__(self):
        self.df = pd.DataFrame()
        self.prev_adi_value = None

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = ("The name accumulation/distribution comes from the idea that during accumulation buyers "
                    "are in control and the price will be bid up through the day,\n"
                    "or will make a recovery if sold down, in either case more often finishing near the day's high than the low."
                    " The opposite applies during distribution.\n\n"
                "Links: \n"
                    "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/AccumulationDistribution.php5\n"
                    "https://www.investopedia.com/terms/a/accumulationdistribution.asp\n"
                    "https://en.wikipedia.org/wiki/Accumulation/distribution_index")
        return info

    def __get_ad_util(self, clv_vol):
        self.prev_adi_value += clv_vol
        return self.prev_adi_value

    def get_value_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df: pandas Dataframe with high, low, close and volume values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding ADI as a new column, preserving the columns which already exists
        """

        df["CLV"] = ((df["close"] - df["low"]) - (df["high"] -
                                                  df["close"])) / (df["high"] - df["low"])
        df["CLV_VOL"] = df["CLV"] * df["volume"]
        self.prev_adi_value = df["CLV_VOL"][0]
        df["ADI"] = df["CLV_VOL"].apply(self.__get_ad_util)
        df = df.drop(["CLV", "CLV_VOL"], axis=1)
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series, volume_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values: 'Low' values.\n
            close_values: 'Close' values.\n
            volume_values: 'Volume' values.\n

        Returns:
            pandas.Series: A pandas Series of ADI values
        """

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

    """
    ATR -> Average True Range

    Average True Range is a volatility indicator which provides degree of price of volatility making use of smoothed moving average of true ranges.

    \nLinks:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/AverageTrueRange.php5\n
        https://www.investopedia.com/terms/a/atr.asp\n
        https://en.wikipedia.org/wiki/Average_true_range
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = ("Average True Range is a volatility indicator which provides degree of price of volatility making use of smoothed moving average of true ranges.\n\n"
                "Links: \n"
                    "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/AverageTrueRange.php5\n"
                    "https://www.investopedia.com/terms/a/atr.asp\n"
                    "https://en.wikipedia.org/wiki/Average_true_range\n"
                )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low and close values\n
            time_period(int): look back period to calculate ATR

        Returns:
            pandas.DataFrame: new pandas dataframe adding ATR as a new column, preserving the columns which already exists
        """

        df["close_prev"] = df["close"].shift(1)
        df["TR"] = df[["high", "close_prev"]].max(
            axis=1) - df[["low", "close_prev"]].min(axis=1)
        df["ATR"] = df["TR"].rolling(window=time_period).mean()

        df = df.drop(["close_prev", "TR"], axis=1)

        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values: 'Low' values.\n
            close_values: 'Close' values.\n
            time_period: Look back time period\n

        Returns:
            pandas.Series: A pandas Series of ATR values
        """

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

class BollingerBands:
    """
    Bollinger Bands Indicator

    Bollinger Bands are the type of indicators which use mean and standard deviation of the movement of the stock to estimate the volatility of the stock\n


    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/BollingerBands.php5\n
        https://www.investopedia.com/terms/b/bollingerbands.asp\n
        https://en.wikipedia.org/wiki/Bollinger_Bands
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """
        info = (
        "Bollinger Bands are the type of indicators which use mean and standard deviation of the movement of the stock to estimate the volatility of the stock\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/BollingerBands.php5\n"
            "https://www.investopedia.com/terms/b/bollingerbands.asp\n"
            "https://en.wikipedia.org/wiki/Bollinger_Bands"
        )
        return info


    def get_value_df(self, df: pd.DataFrame, time_period: int = 20, std_dev_multiplier: int = 2):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n
            time_period(int): look back time period to calculate moving average
            std_dev_multiplier(int): constant value which will be multiplied by standard deviation

        Returns:
            pandas.DataFrame: new pandas dataframe adding bb_upper and bb_lower as two columns, preserving the columns which already exists\n
        """

        df["TYP"] = (df["high"] + df["low"] + df["close"] ) / 3
        df["TYPMA"] = df["TYP"].rolling(window=time_period).mean()
        df["TYPSD"] = df["TYP"].rolling(window=time_period).std()
        df["bb_upper"] = df["TYPMA"] + std_dev_multiplier * df["TYPSD"]
        df["bb_lower"] = df["TYPMA"] - std_dev_multiplier * df["TYPSD"]

        df = df.drop(["TYP", "TYPMA", "TYPSD"], axis=1)

        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series, time_period: int = 20, std_dev_multiplier: int = 2):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period to calculate moving average
            std_dev_multiplier(int): constant value which will be multiplied by standard deviation

        Returns:
            pandas.Series: A tuple containing bb_upper and bb_lower values
        """

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df["TYP"] = (self.df["high"] + self.df["low"] + self.df["close"] ) / 3
        self.df["TYPMA"] = self.df["TYP"].rolling(window=time_period).mean()
        self.df["TYPSD"] = self.df["TYP"].rolling(window=time_period).std()
        bb_upper = self.df["TYPMA"] + std_dev_multiplier * self.df["TYPSD"]
        bb_lower = self.df["TYPMA"] - std_dev_multiplier * self.df["TYPSD"]

        return bb_upper, bb_lower


class CCI:
    """
    CCI -> Commodity Channel Index

    Main intention behind using the commodity channel index is to identify cyclical trends.\n


    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/CommodityChannelIndex.php5\n
        https://www.investopedia.com/terms/c/commoditychannelindex.asp\n
        https://en.wikipedia.org/wiki/Commodity_channel_index
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Main intention behind using the commodity channel index is to identify cyclical trends.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/CommodityChannelIndex.php5\n"
            "https://www.investopedia.com/terms/c/commoditychannelindex.asp\n"
            "https://en.wikipedia.org/wiki/Commodity_channel_index"
        )
        return info




    def get_value_df(self, df: pd.DataFrame, time_period: int = 20, sd_multiplier: float = 0.015):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n
            time_period(int): look back time period\n
            sd_multiplier(float): constant value to be multiplied by standard deviation

        Returns:
            pandas.DataFrame: new pandas dataframe adding CCI as a new column, preserving the columns which already exists\n
        """

        df["TYP"] = (df["high"] + df["low"] + df["close"]) / 3
        df["MA"] = df["TYP"].rolling(window=time_period).mean()
        df["STD"] = df["TYP"].rolling(window=time_period).std()

        df["CCI"] = ((df["TYP"] - df["MA"]) / (sd_multiplier * df["STD"]))

        df = df.drop(["TYP", "MA", "STD"], axis=1)
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series,  time_period: int = 20,  sd_multiplier: float = 0.015):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period\n
            sd_multiplier(float): constant value to be multiplied by standard deviation

        Returns:
            pandas.Series: pandas series of CCI values
        """

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df["TYP"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        self.df["MA"] = self.df["TYP"].rolling(window=time_period).mean()
        self.df["STD"] = self.df["TYP"].rolling(window=time_period).std()

        cci_values = ((self.df["TYP"] - self.df["MA"]) / (sd_multiplier * self.df["STD"]))


        self.df = pd.DataFrame(None)

        return cci_values

class CMF:

    """
    CMF -> Chaikin Money Flow

    Chaikin Money flow is used to measure money flow volume over a certain time periods.\n

    \n Links:
          http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ChaikinMoneyFlow.php5\n
          https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_Money_Flow
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Chaikin Money flow is used to measure money flow volume over a certain time periods.\n "

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ChaikinMoneyFlow.php5\n"
            "https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_Money_Flow"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 20):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, close and volume values\n
            time_period(int): look back period to calculate CMF

        Returns:
            pandas.DataFrame: new pandas dataframe adding CMF as a new column, preserving the columns which already exists
        """

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

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series,  close_values: pd.Series, volume_values: pd.Series, time_period: int = 20):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values: 'Low' values.\n
            close_values: 'Close' values.\n
            volume_levels: 'Volume' values\n
            time_period: Look back time period\n

        Returns:
            pandas.Series: A pandas Series of CMF values
        """

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
    """
    CHO -> Chaikin Oscillators

    Chaikin oscillator is designed to anticipate the directional changes in Accumulation distributin line by measuring the momentum behind the movements.\n

    \n Links:
         http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ChaikinOscillator.php5\n
         https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_oscillator\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Chaikin oscillator is designed to anticipate the directional changes in Accumulation distributin line by measuring the momentum behind the movements.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ChaikinOscillator.php5\n"
            "https://en.wikipedia.org/wiki/Chaikin_Analytics#Chaikin_oscillator\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, short_time_period: int = 3, long_time_period: int = 10):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, close and volume values\n
            short_time_period(int): look back period to calculate short term moving average\n
            long_time_period(int): look back period to calculate long term moving average

        Returns:
            pandas.DataFrame: new pandas dataframe adding CHO as a new column, preserving the columns which already exists
        """

        df["AD"] = ((2 * df["close"] - (df["high"] + df["low"])
                     ) / (df["high"] - df["low"])) * df["volume"]
        df["AD_short"] = df["AD"].ewm(span=short_time_period).mean()
        df["AD_long"] = df["AD"].ewm(span=long_time_period).mean()

        df["CHO"] = df["AD_short"] - df["AD_long"]
        df = df.drop(["AD", "AD_short", "AD_long"], axis=1)
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series,
                       volume_values: pd.Series, short_time_period: int = 3, long_time_period: int = 10):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values: 'Low' values.\n
            close_values: 'Close' values.\n
            volume_levels: 'Volume' values\n
            short_time_period(int): look back period to calculate short term moving average\n
            long_time_period(int): look back period to calculate long term moving average\n

        Returns:
            pandas.Series: A pandas Series of CHO values
        """

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
    """
    CHV -> Chaikin Volatility

    Chaikin Volatility determines the volatility of instrument using percentage change in a moving average of difference between high price and the low price over a specific period of time.\n

    \n Links:
         http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ChaikinVolatility.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Chaikin Volatility determines the volatility of instrument using percentage change in a moving average"
        " of difference between high price and the low price over a specific period of time.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ChaikinVolatility.php5\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 10):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high and low values\n
            time_period(int): look back period to calculate moving average\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding CHV as a new column, preserving existing columns\n
        """

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

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, time_period: int = 10):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values: 'Low' values.\n
            time_period(int): look back period to calculate moving average\n

        Returns:
            pandas.Series: A pandas Series of CHV values
        """

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

class DC:
    """
    DC -> Donchian Channel

    Donchian Channel is used as a volatility indicator by providing a channel of three lines whose contraction and expansion acting as signals.\n

    \n Links:
         https://www.investopedia.com/terms/d/donchianchannels.asp\n
         https://en.wikipedia.org/wiki/Donchian_channel\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Donchian Channel is used as a volatility indicator by providing a channel of three lines whose contraction and expansion acting as signals.\n"

        "\n Links:\n"
            "https://www.investopedia.com/terms/d/donchianchannels.asp\n"
            "https://en.wikipedia.org/wiki/Donchian_channel\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low values\n
            time_period(int): look back time period

        Returns:
            pandas.DataFrame: new pandas dataframe adding dc_upper, dc_middle and dc_lower as three columns, preserving the columns which already exists\n
        """

        df["dc_upper"] = df["high"].rolling(window=time_period).max()
        df["dc_lower"] = df["low"].rolling(window=time_period).min()
        df["dc_middle"] = (df["dc_upper"] + df["dc_lower"]) / 2

        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            time_period(int): look back time period


        Returns:
            pandas.Series: A tuple containing dc_upper, dc_middle and dc_lower values
        """

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values
        })
        self.df["dc_upper"] = self.df["high"].rolling(window=time_period).max()
        self.df["dc_lower"] = self.df["low"].rolling(window=time_period).min()
        self.df["dc_middle"] = (self.df["dc_upper"] + self.df["dc_lower"]) / 2

        dc_middle, dc_upper, dc_lower = self.df["dc_middle"], self.df["dc_upper"], self.df["dc_lower"]

        self.df = pd.DataFrame(None)

        return dc_upper, dc_middle, dc_lower





class DPO:
    """
    DPO -> Detrended Price Oscillator

    Detrend Price Oscillator tries to eliminates long term trends in order to easily identify small term trends.\n

    \n Links:
         http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/DetrendedPriceOscillator.php5\n
         https://www.investopedia.com/terms/d/detrended-price-oscillator-dpo.asp\n
         https://en.wikipedia.org/wiki/Detrended_price_oscillator\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Detrend Price Oscillator tries to eliminates long term trends in order to easily identify small term trends.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/DetrendedPriceOscillator.php5\n"
            "https://www.investopedia.com/terms/d/detrended-price-oscillator-dpo.asp\n"
            "https://en.wikipedia.org/wiki/Detrended_price_oscillator\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 20):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding DPO as a new column, preserving the columns which already exists\n
        """

        df["close_prev"] = df["close"].shift(int(time_period / 2 + 1))
        df["SMA"] = df["close"].rolling(window=time_period).mean()

        df["DPO"] = df["close_prev"] - df["SMA"]

        df = df.drop(["close_prev", "SMA"], axis=1)
        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 20):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values.\n
            time_period(int): look back period to calculate moving average\n

        Returns:
            pandas.Series: A pandas Series of DPO values
        """

        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(
            int(time_period / 2 + 1))
        self.df["SMA"] = self.df["close"].rolling(window=time_period).mean()

        dpo_values = self.df["close_prev"] - self.df["SMA"]
        self.df = pd.DataFrame(None)
        return dpo_values


class EMV:
    """
    EMV -> Ease of Movement

    Ease of movement tries to identify amount of volume needed to move prices.\n

    \n Links:
         http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/EaseOfMovement.php5\n
         https://www.investopedia.com/terms/e/easeofmovement.asp\n
         https://en.wikipedia.org/wiki/Ease_of_movement\n
    """
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Ease of movement tries to identify amount of volume needed to move prices.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/EaseOfMovement.php5\n"
            "https://www.investopedia.com/terms/e/easeofmovement.asp\n"
            "https://en.wikipedia.org/wiki/Ease_of_movement\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, volume_divisor: int = 1000000, need_moving_average: bool = True, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low and volume values\n
            volume_divisor(int): arbitrary divisor value required in the calculation of EMV\n
            need_moving_average(bool): if True the moving avearge of the calculated values are returned
            time_period(int): look back time period\n


        Returns:
            pandas.DataFrame: new pandas dataframe adding EMV as a new column, preserving the columns which already exists\n
        """

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

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, volume_values: pd.Series, volume_divisor: int = 1000000,
                       need_moving_average: bool = True, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values(pandas.Series): 'Low' values.\n
            volume_values(pandas.Series): 'Volume' values.\n
            volume_divisor(int): arbitrary divisor value required in the calculation of EMV\n
            need_moving_average(bool): if True the moving avearge of the calculated values are returned\n
            time_period(int): look back time period\n

        Returns:
            pandas.Series: A pandas Series of EMV values
        """

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
    """
    EMA -> Exponential Moving Average

    Exponential Moving Average is a type of moving average which puts more weightage to the recent points, where as moving average puts same weightage all the points in consideration.\n

    \n Links:
         http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MovingAverage.php5\n
         https://www.investopedia.com/terms/e/ema.asp\n
         https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average\n
    """
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Exponential Moving Average is a type of moving average which puts more weightage to the recent points, where as moving average puts same weightage all the points in consideration.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MovingAverage.php5\n"
            "https://www.investopedia.com/terms/e/ema.asp\n"
            "https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 21):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period\n


        Returns:
            pandas.DataFrame: new pandas dataframe adding EMA as a new column, preserving the columns which already exists\n
        """

        df["EMA"] = df["close"].ewm(span=time_period).mean()
        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 21):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values.\n
            time_period(int): look back time period\n

        Returns:
            pandas.Series: A pandas Series of EMA values
        """

        self.df["close"] = close_values
        ema_values = self.df["close"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)
        return ema_values


class FI:
    """
    FI -> Force Index

    Force index tries to determine the amount of power used to move the price of an asset.\n

    \n Links:
         http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ForceIndex.php5\n
         https://www.investopedia.com/terms/f/force-index.asp\n
         https://en.wikipedia.org/wiki/Force_index\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Force index tries to determine the amount of power used to move the price of an asset.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/ForceIndex.php5\n"
            "https://www.investopedia.com/terms/f/force-index.asp\n"
            "https://en.wikipedia.org/wiki/Force_index\n"
        )
        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close and volume values\n
            time_period(int): look back time period\n


        Returns:
            pandas.DataFrame: new pandas dataframe adding FI as a new column, preserving the columns which already exists\n
        """

        df["close_prev"] = df["close"].shift(1)
        df["FI"] = (df["close"] -
                    df["close_prev"]) * df["volume"]
        df["FI"] = df["fi"].ewm(span=time_period).mean()
        self.df = pd.DataFrame(None)
        df = df.drop(["close_prev"], axis=1)
        return df

    def get_value_list(self, close_values: pd.Series, volume_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values.\n
            volume_values(pandas.Series): 'Volume' values.\n
            time_period(int): look back time period\n

        Returns:
            pandas.Series: A pandas Series of FI values
        """

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

class KC:
    """
    KC -> Keltner Channel

    Keltner Channel is a volatility indicator which provides a channel of 3 lines whose contraction and expansion are used as signals.\n

    \n Links:
         https://www.investopedia.com/terms/k/keltnerchannel.asp\n
         https://en.wikipedia.org/wiki/Keltner_channel\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """
        info = (
        "Keltner Channel is a volatility indicator which provides a channel of 3 lines whose contraction and expansion are used as signals.\n"

        "\n Links:\n"
            "https://www.investopedia.com/terms/k/keltnerchannel.asp\n"
            "https://en.wikipedia.org/wiki/Keltner_channel\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 20, atr_time_period: int = 14, atr_multiplier: int = 2):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average
            atr_time_period(int): time period to calculate average true range
            atr_multiplier(int): constant value which will be multiplied by average true range

        Returns:
            pandas.DataFrame: new pandas dataframe adding kc_upper, kc_middle and kc_lower as three columns, preserving the columns which already exists\n
        """

        df["kc_middle"] = df["close"].ewm(span=time_period).mean()
        df = ATR().get_value_df(df, atr_time_period)

        df["kc_upper"] = df["kc_middle"] + atr_multiplier * df["ATR"]
        df["kc_lower"] = df["kc_middle"] - atr_multiplier * df["ATR"]

        df = df.drop(["ATR"], axis=1)

        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 20, atr_time_period: int = 14, atr_multiplier: int = 2):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period to calculate moving average
            atr_time_period(int): time period to calculate average true range
            atr_multiplier(int): constant value which will be multiplied by average true range

        Returns:
            pandas.Series: A tuple containing kc_upper, kc_middle and kc_lower values
        """

        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["kc_middle"] = self.df["close"].ewm(span=time_period).mean()
        self.df = ATR().get_value_df(self.df, atr_time_period)

        self.df["kc_upper"] = self.df["kc_middle"] + atr_multiplier * self.df["ATR"]
        self.df["kc_lower"] = self.df["kc_middle"] - atr_multiplier * self.df["ATR"]

        kc_middle, kc_upper, kc_lower = self.df["kc_middle"], self.df["kc_upper"], self.df["kc_lower"]

        self.df = pd.DataFrame(None)

        return kc_upper, kc_middle, kc_lower

class KST:
    """
    KST -> KST Oscillator

    KST oscillator is a momentum indicator which makes it easy to interpret rate of change indicator.\n

    \n Links:
         https://www.investopedia.com/terms/k/know-sure-thing-kst.asp\n
         https://en.wikipedia.org/wiki/KST_oscillator\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "KST oscillator is a momentum indicator which makes it easy to interpret rate of change indicator.\n"

        "\n Links:\n"
            "https://www.investopedia.com/terms/k/know-sure-thing-kst.asp\n"
            "https://en.wikipedia.org/wiki/KST_oscillator\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, roc1: int = 10, roc1_ma_period: int = 10, roc2: int = 15, roc2_ma_period: int = 10,
                    roc3: int = 20, roc3_ma_period: int = 10, roc4: int = 30, roc4_ma_period: int = 15):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            roc1(int): ROC Calculator - 1\n
            roc1_ma_period(int): Smoothing constant for roc1\n
            roc2(int): ROC Calculator - 2\n
            roc2_ma_period(int): Smoothing constant for roc2\n
            roc3(int): ROC Calculator - 3\n
            roc3_ma_period(int): Smoothing constant for roc3\n
            roc4(int): ROC Calculator - 4\n
            roc4_ma_period(int): Smoothing constant for roc4\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding KST as a new column, preserving the columns which already exists\n
        """
        roc = ROC()
        df["RCMA1"] = roc.get_value_list(df["close"], time_period=roc1)
        df["RCMA1"] = df["RCMA1"].rolling(window=roc1_ma_period).mean()

        df["RCMA2"] = roc.get_value_list(df["close"], time_period=roc2)
        df["RCMA2"] = df["RCMA2"].rolling(window=roc2_ma_period).mean()
        df["RCMA3"] = roc.get_value_list(df["close"], time_period=roc3)
        df["RCMA3"] = df["RCMA3"].rolling(window=roc3_ma_period).mean()
        df["RCMA4"] = roc.get_value_list(df["close"], time_period=roc4)
        df["RCMA4"] = df["RCMA4"].rolling(window=roc4_ma_period).mean()

        df["KST"] = (df["RCMA1"] * 1) + (df["RCMA2"] * 2) + (df["RCMA3"] * 3) + (df["RCMA4"] * 4)
        df = df.drop(["RCMA1", "RCMA2", "RCMA3", "RCMA4"], axis=1)
        return df


    def get_value_list(self, close_values: pd.Series, roc1: int = 10, roc1_ma_period: int = 10, roc2: int = 15, roc2_ma_period: int = 10,
                     roc3: int = 20, roc3_ma_period: int = 10, roc4: int = 30, roc4_ma_period: int = 15):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            roc1(int): ROC Calculator - 1\n
            roc1_ma_period(int): Smoothing constant for roc1\n
            roc2(int): ROC Calculator - 2\n
            roc2_ma_period(int): Smoothing constant for roc2\n
            roc3(int): ROC Calculator - 3\n
            roc3_ma_period(int): Smoothing constant for roc3\n
            roc4(int): ROC Calculator - 4\n
            roc4_ma_period(int): Smoothing constant for roc4\n

        Returns:
            pandas.Series: A pandas Series of KST values
        """

        self.df = pd.DataFrame({
            "close": close_values
        })
        roc = ROC()
        self.df["RCMA1"] = roc.get_value_list(self.df["close"], time_period=roc1)
        self.df["RCMA1"] = self.df["RCMA1"].rolling(window=roc1_ma_period).mean()

        self.df["RCMA2"] = roc.get_value_list(self.df["close"], time_period=roc2)
        self.df["RCMA2"] = self.df["RCMA2"].rolling(window=roc2_ma_period).mean()
        self.df["RCMA3"] = roc.get_value_list(self.df["close"], time_period=roc3)
        self.df["RCMA3"] = self.df["RCMA3"].rolling(window=roc3_ma_period).mean()
        self.df["RCMA4"] = roc.get_value_list(self.df["close"], time_period=roc4)
        self.df["RCMA4"] = self.df["RCMA4"].rolling(window=roc4_ma_period).mean()

        kst_values = (self.df["RCMA1"] * 1) + (self.df["RCMA2"] * 2) + (self.df["RCMA3"] * 3) + (self.df["RCMA4"] * 4)


        self.df = pd.DataFrame(None)

        return kst_values





class MI:
    """
    MI -> Mass Index

    Mass index tries to determine the range of high and low values over a specified period of time\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MassIndex.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Mass index tries to determine the range of high and low values over a specified period of time.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MassIndex.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 25, ema_time_period: int = 9):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high and low values\n
            time_period(int): look back time period to calculate the sum\n
            ema_time_period(int): look back time period to calculate the exponential moving average\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding MI as a new column, preserving the columns which already exists\n
        """

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

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, time_period: int = 25, ema_time_period: int = 9):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values(pandas.Series): 'Low' values.\n
            time_period(int): look back time period to calculate the sum\n
            ema_time_period(int): look back time period to calculate the exponential moving average\n

        Returns:
            pandas.Series: A pandas Series of MI values
        """

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
    """
    MED -> Median Price

    Median Price calculates a mid point of a  price range. \n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MedianPrice.php5\n
    """


    def __init__(self):
        self.df = None

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Median Price calculates a mid point of a price range. \n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MedianPrice.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high and low values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding MED as a new column, preserving the columns which already exists\n
        """

        df["MED"] = (df["high"] + df["low"]) / 2
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values.\n
            low_values(pandas.Series): 'Low' values.\n
        Returns:
            pandas.Series: A pandas Series of MED values
        """

        return (high_values + low_values) / 2


class MOM:
    """
    MOM -> Momentum

    Momentum helps to determine the price changes from one period to another. \n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Momentum.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Momentum helps to determine the price changes from one period to another.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Momentum.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 1):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period.\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding MOM as a new column, preserving the columns which already exists\n
        """
        df["close_prev"] = df["close"].shift(time_period)
        df["MOM"] = df["close"] - df["close_prev"]
        self.df = pd.DataFrame(None)

    def get_value_list(self, close_values: pd.Series, time_period: int = 1):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period\n

        Returns:
            pandas.Series: A pandas Series of MOM values
        """

        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(time_period)
        self.df["MOM"] = self.df["close"] - self.df["close_prev"]
        mom_values = self.df["MOM"]
        self.df = pd.DataFrame(None)
        return mom_values


class MFI:
    """
    MFI -> Money Flow Index


    Money flow index uses price and volume data to for identifying overbought and oversold signals of an asset
    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MoneyFlowIndex.php5\n
        https://www.investopedia.com/terms/m/mfi.asp
        https://en.wikipedia.org/wiki/Money_flow_index
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Momentum helps to determine the price changes from one period to another.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MoneyFlowIndex.php5\n"
            "https://www.investopedia.com/terms/m/mfi.asp\n"
            "https://en.wikipedia.org/wiki/Money_flow_index\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, close and volume values\n
            time_period(int): look back time period.\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding MFI as a new column, preserving the columns which already exists\n
        """
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

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series, volume_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n
            volume_values(pandas.Series): 'Volume' values\n
            time_period(int): look back time period\n

        Returns:
            pandas.Series: A pandas Series of MFI values
        """
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
    """
    MACD -> Moving Average Convergence Divergence

    Moving Average Convergence is a trend following momentum indicator that shows a relationship between two moving averages of an asset.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MovingAverageConvergenceDivergence.php5\n
        https://www.investopedia.com/terms/m/macd.asp\n
        https://en.wikipedia.org/wiki/MACD\n
    """


    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Moving Average Convergence is a trend following momentum indicator that shows a relationship between two moving averages of an asset.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MovingAverageConvergenceDivergence.php5\n"
            "https://www.investopedia.com/terms/m/macd.asp\n"
            "https://en.wikipedia.org/wiki/MACD\n"
        )

        return info


    def get_value_df(self, df: pd.DataFrame, short_time_period: int = 12, long_time_period: int = 26,
                     need_signal: bool = True, signal_time_period: int = 9):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            short_time_period(int): short term look back time period.\n
            long_time_period(int): long term look back time period.\n
            need_signal(bool): if True MACD signal line is added as a new column to the returning pandas dataframe.\n
            signal_time_period(int): look back period to calculate signal line\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding MACD and MACD_signal_line(if required) as new column/s, preserving the columns which already exists\n
        """

        df["LONG_EMA"] = df["close"].ewm(span=long_time_period).mean()
        df["SHORT_EMA"] = df["close"].ewm(span=short_time_period).mean()

        df["MACD"] = df["SHORT_EMA"] - df["LONG_EMA"]
        if need_signal:
            df["MACD_signal_line"] = df["MACD"].ewm(
                span=signal_time_period).mean()

        df = df.drop(["LONG_EMA", "SHORT_EMA"], axis=1)
        return df

    def get_value_list(self, close_values: pd.Series, short_time_period: int = 12, long_time_period: int = 26, need_signal: bool = True, signal_time_period: int = 9):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            short_time_period(int): short term look back time period.\n
            long_time_period(int): long term look back time period.\n
            need_signal(bool): if True MACD signal line is also returned along with MACD line\n
            signal_time_period(int): look back period to calculate signal line\n

        Returns:
            pandas.Series: A tuple containing MACD and MACD_signal_line(if required)
        """

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
        """
        Provides basic information about the indicator
        """

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
    """
    NVI -> Negative Volume Index

    Negative Volume Index helps in identifying trends and reversals.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/NegativeVolumeIndex.php5\n
        https://www.investopedia.com/terms/n/nvi.asp\n
        https://en.wikipedia.org/wiki/Negative_volume_index\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)
        self.prev_nvi_value = 1000

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Negative Volume Index helps in identifying trends and reversals.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/NegativeVolumeIndex.php5\n"
            "https://www.investopedia.com/terms/n/nvi.asp\n"
            "https://en.wikipedia.org/wiki/Negative_volume_index\n"
        )

        return info

    def __get_nvi_util(self, df_row):
        if df_row["volume"] < df_row["volume_prev"]:
            self.prev_nvi_value = self.prev_nvi_value * \
                (1 + ((df_row["close"] - df_row["close_prev"]) /
                      df_row["close_prev"]))
        return self.prev_nvi_value

    def get_value_df(self, df: pd.DataFrame, start_value: int = 1000):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close and volume values\n
            start_value(int): arbitrary starting value to calculate NVI

        Returns:
            pandas.DataFrame: new pandas dataframe adding NVI as new column, preserving the columns which already exists\n
        """

        self.prev_nvi_value = start_value

        df["close_prev"] = df["close"].shift(1)
        df["volume_prev"] = df["volume"].shift(1)

        df["NVI"] = df.apply(self.__get_nvi_util, axis=1)

        df = df.drop(["close_prev", "volume_prev"], axis=1)

        return df

    def get_value_list(self, close_values: pd.Series, volume_values: pd.Series, start_value: int = 1000):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            volume_values(pands.Series): 'Volume' values\n
            start_value(int): arbitrary starting value to calculate NVI

        Returns:
            pandas.Series: A pandas Series of NVI values
        """
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
    """
    OBV -> On Balance Volume

    On Balance Volume provides the signal whether the volume is flowing in or out of a given security.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/OnBalanceVolume.php5\n
        https://www.investopedia.com/terms/o/onbalancevolume.asp\n
        https://en.wikipedia.org/wiki/On-balance_volume\n
    """

    def __init__(self):
        self.df = None
        self.obv_value = None

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "On Balance Volume provides the signal whether the volume is flowing in or out of a given security.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/OnBalanceVolume.php5\n"
            "https://www.investopedia.com/terms/o/onbalancevolume.asp\n"
            "https://en.wikipedia.org/wiki/On-balance_volume\n"
        )

        return info

    def __get_obv_util(self, df_row):
        if df_row["close"] > df_row["close_prev"]:
            self.obv_value = self.obv_value + df_row["volume"]
        elif df_row["close"] < df_row["close_prev"]:
            self.obv_value = self.obv_value - df_row["volume"]
        return self.obv_value

    def get_value_df(self, df: pd.DataFrame):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close and volume values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding OBV as new column, preserving the columns which already exists\n
        """

        self.obv_value = df.iloc[0]["volume"]
        df["close_prev"] = df["close"].shift(1)
        df["OBV"] = df.apply(self.__get_obv_util, axis=1)

        df = df.drop(["close_prev"], axis=1)
        return df

    def get_value_list(self, close_values: pd.Series, volume_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            volume_values(pands.Series): 'Volume' values\n

        Returns:
            pandas.Series: A pandas Series of OBV values
        """

        self.obv_value = df.iloc[0]["volume"]
        self.df = pd.DataFrame({
            "close": close_values,
            "volume": volume_values
        })

        self.df["close_prev"] = self.df["close"].shift(1)
        obv_values = self.df.apply(self.__get_obv_util, axis=1)

        return obv_values


class PositiveDirectionIndicator:
    """
    MOM -> Momentum
    """
    """
    Returns a dataframe adding MOM as a new column
    """
    """
    Returns a series of MOM values
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

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

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, time_period: int = 14):
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
        """
        Provides basic information about the indicator
        """

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

    def get_value_list(self, close_values: pd.Series, volume_values: pd.Series, starting_value: int = 100):
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
    """
    PVT -> Price Volume Trend

    Price Volume Trend helps in identifying trend by using cumulative volume adjusted by change in price\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/PriceAndVolumeTrend.php5\n
        https://www.investopedia.com/terms/v/vptindicator.asp\n
        https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend\n
    """

    def __init__(self):
        self.df = None
        self.pvt_value = None

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Price Volume Trend helps in identifying trend by using cumulative volume adjusted by change in price.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/PriceAndVolumeTrend.php5\n"
            "https://www.investopedia.com/terms/v/vptindicator.asp\n"
            "https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend\n"
        )

        return info

    def __get_pvt_util(self, df_row):
        if not np.isnan(df_row["close"]) and not np.isnan(df_row["close_prev"]):
            self.pvt_value = self.pvt_value + \
                (((df_row["close"] - df_row["close_prev"]) /
                  df_row["close_prev"]) * df_row["volume"])
        return self.pvt_value

    def get_value_df(self, df: pd.DataFrame):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close and volume values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding PVT as new column, preserving the columns which already exists\n
        """

        self.pvt_value = df.iloc[0]["volume"]
        df["close_prev"] = df["close"].shift(1)
        df["PVT"] = df.apply(self.__get_pvt_util, axis=1)

        return df

    def get_value_list(self, close_values: pd.Series, volume_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            volume_values(pands.Series): 'Volume' values\n

        Returns:
            pandas.Series: A pandas Series of PVT values
        """

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
    """
    PC -> Price Channels

    Price channels forms a boundary and between them the close price of an asset is oscillating.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/PriceChannels.php5\n
        https://www.investopedia.com/terms/p/price-channel.asp\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Price channels forms a boundary and between them the close price of an asset is oscillating.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/PriceChannels.php5\n"
            "https://www.investopedia.com/terms/p/price-channel.asp\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, percent_value: int = 6, time_period: int = 21):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            percent_value(int): value to calculate the percentage of close value to create the boundary\n
            time_period(int): look back time period to calculate moving average\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding PC as new column, preserving the columns which already exists\n
        """

        df["EMA_FOR_PC"] = df["close"].ewm(span=ema_period).mean()

        df["PC_upper"] = df["EMA_FOR_PC"] * (1 + (percent_value / 100))
        df["PC_lower"] = df["EMA_FOR_PC"] * (1 - (percent_value / 100))

        df = df.drop(["EMA_FOR_PC"], axis=1)

        return df

    def get_value_list(self, close_values: pd.Series, percent_value: int = 6, ema_period: int = 21):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            percent_value(int): value to calculate the percentage of close value to create the boundary\n
            time_period(int): look back time period to calculate moving average\n

        Returns:
            pandas.Series: A tuple containing PC_upper and PC_lower values
        """

        self.df = pd.DataFrame({
            "close": close_values
        })

        self.df["EMA_FOR_PC"] = self.df["close"].ewm(span=ema_period).mean()

        pc_upper_values = self.df["EMA_FOR_PC"] * (1 + (percent_value / 100))
        pc_lower_values = self.df["EMA_FOR_PC"] * (1 - (percent_value / 100))

        self.df = pd.DataFrame(None)
        return pc_upper_values, pc_lower_values


class PO:
    """
    PO -> Price Oscillator

    Price oscillator is a momentum osciallator which shows a difference between two moving averages.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/PriceOscillator.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Price oscillator is a momentum osciallator which shows a difference between two moving averages.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/PriceOscillator.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, short_time_period: int = 9, long_time_period: int = 26):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            short_time_period(int): look back time period to calculate short term moving average\n
            long_time_period(int): look back time period to calculate long term moving average\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding PO as new column, preserving the columns which already exists\n
        """

        df["Short_EMA"] = df["close"].ewm(
            span=short_ema_period).mean()

        df["Short_EMA"].iloc[:short_ema_period] = np.nan

        df["Long_EMA"] = df["close"].ewm(span=long_ema_period).mean()
        df["Long_EMA"].iloc[:long_ema_period] = np.nan

        df["PO"] = ((df["Short_EMA"] - df["Long_EMA"]) /
                    df["Long_EMA"]) * 100
        df = df.drop(["Short_EMA", "Long_EMA"], axis=1)
        return df

    def get_value_list(self, close_values: pd.Series, short_ema_period: int = 9, long_ema_period: int = 26):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            short_time_period(int): look back time period to calculate short term moving average\n
            long_time_period(int): look back time period to calculate long term moving average\n

        Returns:
            pandas.Series: A pandas Series of PO values
        """

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
    """
    ROC -> Rate Of Change

    Rate of change helps in calculation of speed of ascent or descent.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/RateOfChange.php5\n
        https://www.investopedia.com/terms/p/pricerateofchange.asp\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Rate of change helps in calculation of speed of ascent or descent.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/RateOfChange.php5\n"
            "https://www.investopedia.com/terms/p/pricerateofchange.asp\n"
        )

        return info


    def get_value_df(self, df: pd.DataFrame, time_period: int = 12):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate previous close \n

        Returns:
            pandas.DataFrame: new pandas dataframe adding ROC as new column, preserving the columns which already exists\n
        """

        df["close_prev"] = df["close"].shift(time_period)

        df["ROC"] = (df["close"] - df["close_prev"]) / \
            df["close_prev"] * 100

        df = df.drop(["close_prev"], axis=1)

        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 12):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period \n

        Returns:
            pandas.Series: A pandas Series of ROC values
        """

        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["close_prev"] = self.df["close"].shift(time_period)

        roc_values = (self.df["close"] - self.df["close_prev"]) / \
            self.df["close_prev"] * 100
        self.df = pd.DataFrame(None)
        return roc_values


class RSI:
    """
    RSI -> Relative Strength Index

    Relative Strength Index is used to generate oversold and overbought signals.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/RelativeStrengthIndex.php5\n
        https://www.investopedia.com/terms/r/rsi.asp\n
        https://en.wikipedia.org/wiki/Relative_strength_index\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Relative Strength Index is used to generate oversold and overbought signals.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/RelativeStrengthIndex.php5\n"
            "https://www.investopedia.com/terms/r/rsi.asp\n"
            "https://en.wikipedia.org/wiki/Relative_strength_index\n"
        )

        return info

    def get_value_df(self, df, time_period=14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average \n

        Returns:
            pandas.DataFrame: new pandas dataframe adding RSI as new column, preserving the columns which already exists\n
        """

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

    def get_value_list(self, close_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period \n

        Returns:
            pandas.Series: A pandas Series of RSI values
        """

        self.df["close"] = close_values
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["GAIN"] = 0.0
        self.df["LOSS"] = 0.0

        self.df.loc[self.df["close"] > self.df["close_prev"],
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
    """
    SMA -> Simple Moving Avearge

    Simple Moving Average is an arithmetic moving average which is calculated by taking the sum of values from recent time periods and then divided by number of time periods.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MovingAverage.php5\n
        https://www.investopedia.com/terms/s/sma.asp\n
        https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average\n
    """


    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Simple Moving Average is an arithmetic moving average which is calculated by taking the sum of values from recent time periods and then divided by number of time periods.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/MovingAverage.php5\n"
            "https://www.investopedia.com/terms/s/sma.asp\n"
            "https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 21):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average \n

        Returns:
            pandas.DataFrame: new pandas dataframe adding SMA as new column, preserving the columns which already exists\n
        """

        df["SMA"] = df["close"].rolling(window=time_period).mean()
        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 21):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period \n

        Returns:
            pandas.Series: A pandas Series of SMA values
        """

        self.df["close"] = close_values
        sma_values = self.df["close"].rolling(window=time_period).mean()
        self.df = pd.DataFrame(None)
        return sma_values


class VLT:
    """
    VLT -> Volatility

    Standard Deviation, variance and volatility are used to evaluate the volatility in the movement of the stock.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Volatility.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Standard Deviation, variance and volatility are used to evaluate the volatility in the movement of the stock.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Volatility.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 21, need_variance: bool = True, need_deviation: bool = True):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period to calculate moving average \n
            need_variance(bool): if True variance will be added as a new column to the returning dataframe\n
            need_deviation(bool): if True deviation will be added as a new column to the returning dataframe\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding VLT, SV(if required), SD(if required) as new column/s, preserving the columns which already exists\n
        """

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

    def get_value_list(self, close_values: pd.Series, time_period: int = 21, need_variance: bool = True, need_deviation: bool = True):
        """
        Returns a series of SMA values
        """"""
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period to calculate moving average \n
            need_variance(bool): if True variance will be added as a new column to the returning dataframe\n
            need_deviation(bool): if True deviation will be added as a new column to the returning dataframe\n

        Returns:
            pandas.Series: A tuple containing Volatility, variance(if required), deviation(if required) values
        """

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
    """
    StochasticKAndD -> Stochastic K and StochasticD

    Stochastic Oscillator is a momentum indicator comparing a particular price to a range of prices over specific period of time.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Stochastic.php5\n
        https://www.investopedia.com/terms/s/stochasticoscillator.asp\n
        https://en.wikipedia.org/wiki/Stochastic_oscillator\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Stochastic Oscillator is a momentum indicator comparing a particular price to a range of prices over specific period of time.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/Volatility.php5\n"
            "https://www.investopedia.com/terms/s/stochasticoscillator.asp\n"
            "https://en.wikipedia.org/wiki/Stochastic_oscillator\n"
        )

        return info


    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low and close values\n
            time_period(int): look back time period \n

        Returns:
            pandas.DataFrame: new pandas dataframe adding stoc_d and stoc_k as new columns, preserving the columns which already exists\n
        """

        df["highest high"] = df["high"].rolling(
            window=time_period).max()
        df["lowest low"] = df["low"].rolling(
            window=time_period).min()
        df["stoc_k"] = 100 * ((df["close"] - df["lowest low"]) /
                              (df["highest high"] - df["lowest low"]))
        df["stoc_d"] = df["stoc_k"].rolling(window=3).mean()

        df = df.drop(["highest high", "lowest low"], axis=1)
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period \n

        Returns:
            pandas.Series:A tuple containing stoch_k and stoc_d values
        """

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
    """
    Trix -> Triple exponential moving average

    Trix is triple exponential moving average, can be used as both oscillator and momentum indicator.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TRIX.php5\n
        https://www.investopedia.com/terms/t/trix.asp\n
        https://en.wikipedia.org/wiki/Trix_(technical_analysis)\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Trix is triple exponential moving average, can be used as both oscillator and momentum indicator.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TRIX.php5\n"
            "https://www.investopedia.com/terms/t/trix.asp\n"
            "https://en.wikipedia.org/wiki/Trix_(technical_analysis)\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period \n

        Returns:
            pandas.DataFrame: new pandas dataframe adding Trix as new column, preserving the columns which already exists\n
        """

        df["EMA1"] = df["close"].ewm(span=time_period).mean()
        df["EMA2"] = df["EMA1"].ewm(span=time_period).mean()
        df["EMA3"] = df["EMA2"].ewm(span=time_period).mean()
        df["EMA_prev"] = df["EMA3"].shift(1)

        df["TRIX"] = (df["EMA3"] - df["EMA_prev"]) / \
            df["EMA_prev"] * 100
        df = df.drop(["EMA1", "EMA2", "EMA3", "EMA_prev"], axis=1)
        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period \n

        Returns:
            pandas.Series: A pandas Series of Trix values
        """

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
    """
    TR -> True Range

    True range is an essential component of determination of average true range.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TrueRange.php5\n
        https://www.investopedia.com/terms/a/atr.asp\n
        https://en.wikipedia.org/wiki/Average_true_range\n
    """
    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "True range is an essential component of determination of average true range.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TrueRange.php5\n"
            "https://www.investopedia.com/terms/a/atr.asp\n"
            "https://en.wikipedia.org/wiki/Average_true_range\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding TR as new column, preserving the columns which already exists\n
        """

        df["prev_close"] = df["close"].shift(1)
        df["H-L"] = abs(df["high"] - df["low"])
        df["H-CP"] = abs(df["high"] - df["prev_close"])
        df["L-CP"] = abs(df["low"] - df["prev_close"])
        df['TR'] = df[["H-L", "H-CP", "L-CP"]].max(axis=1)

        df = df.drop(["prev_close", "H-L", "H-CP", ], axis=1)
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n

        Returns:
            pandas.Series: A pandas Series of TR values
        """

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


class TSI:
    """
    TSI -> True Strength Index


    True Strength Index is a momentum indicator which is useful in identifying oversold and overbough conditions.\n

    \n Links:
        https://www.investopedia.com/terms/t/tsi.asp\n
        https://en.wikipedia.org/wiki/True_strength_index\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "True Strength Index is a momentum indicator which is useful in identifying oversold and overbough conditions.\n"

        "\n Links:\n"
            "https://www.investopedia.com/terms/t/tsi.asp\n"
            "https://en.wikipedia.org/wiki/True_strength_index\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period1: int = 25, time_period2: int = 13):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period1(int): time period to calculate moving average of price change
            time_period2(int): time period to calcualte moving average of prior moving average

        Returns:
            pandas.DataFrame: new pandas dataframe adding TSI as a new column, preserving the columns which already exists\n
        """
        df["close_prev"] = df["close"].shift(1)
        df["PC"] = df["close"] - df["close_prev"]
        df["PCs"] = df["PC"].ewm(span=time_period1).mean()
        df["PCDs"] = df["PCs"].ewm(span=time_period2).mean()

        df["APC"] = abs(df["close"] - df["close_prev"])
        df["APCs"] = df["APC"].ewm(span=time_period1).mean()
        df["APCDs"] = df["APCs"].ewm(span=time_period2).mean()

        df["TSI"] = df["PCDs"] / df["APCDs"] * 100

        df["TSI"].iloc[:time_period1+time_period2] = np.nan

        df = df.drop(["close_prev", "PC", "PCs", "PCDs", "APC", "APCs", "APCDs"], axis = 1)

        return df



    def get_value_list(self, close_values: pd.Series, time_period1: int = 25, time_period2: int = 13):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period1(int): time period to calculate moving average of price change
            time_period2(int): time period to calcualte moving average of prior moving average

        Returns:
            pandas.Series: A pandas Series of TSI values
        """

        self.df = pd.DataFrame({
            "close": close_values
        })
        self.df["close_prev"] = self.df["close"].shift(1)
        self.df["PC"] = self.df["close"] - self.df["close_prev"]
        self.df["PCs"] = self.df["PC"].ewm(span=time_period1).mean()
        self.df["PCDs"] = self.df["PCs"].ewm(span=time_period2).mean()

        self.df["APC"] = abs(self.df["close"] - self.df["close_prev"])
        self.df["APCs"] = self.df["APC"].ewm(span=time_period1).mean()
        self.df["APCDs"] = self.df["APCs"].ewm(span=time_period2).mean()

        tsi_values = self.df["PCDs"] / self.df["APCDs"] * 100

        tsi_values[: time_period1 + time_period2] = np.nan

        self.df = pd.DataFrame(None)

        return tsi_values




class TYP:
    """
    TYP -> Typical Price

    Typical Price is an average of low, high and close. It is used as an alternative to close price.\n
    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TypicalPrice.php5\n
    """

    def __init__(self):
        self.df = None

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Typical Price is an average of low, high and close. It is used as an alternative to close price.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TypicalPrice.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding TYP as new column, preserving the columns which already exists\n
        """

        df["TYP"] = (df["high"] + df["low"] + df["close"]) / 3

        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n

        Returns:
            pandas.Series: A pandas Series of TYP values
        """

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        typ_values = (self.df["high"] + self.df["low"] + self.df["close"]) / 3

        self.df = pd.DataFrame(None)
        return typ_values


class VHF:
    """
    VHF -> Vertical Horizontal Filter

    VHF is an indicator which is used in identifying trend activity.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TypicalPrice.php5\n
    """


    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "VHF is an indicator which is used in identifying trend activity.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/TypicalPrice.php5\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 28):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with close values\n
            time_period(int): look back time period\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding VHF as new column, preserving the columns which already exists\n
        """

        df["PC"] = df["close"].shift(1)
        df["DIF"] = abs(df["close"] - df["PC"])

        df["HC"] = df["close"].rolling(window=time_period).max()
        df["LC"] = df["close"].rolling(window=time_period).min()

        df["HC-LC"] = abs(df["HC"] - df["LC"])

        df["DIF"] = df["DIF"].rolling(window=time_period).sum()

        df["VHF"] = df["HC-LC"] / df["DIF"]

        df = df.drop(["PC", "DIF", "HC", "LC", "HC-LC"], axis=1)

        return df

    def get_value_list(self, close_values: pd.Series, time_period: int = 28):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period

        Returns:
            pandas.Series: A pandas Series of VHF values
        """

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

class VI:
    """
    VI -> Vortex Indicator

    Vortex indicator is used to identify the start of the new trend or the continuation of existing trends.\n

    \n Links:
        https://www.investopedia.com/terms/v/vortex-indicator-vi.asp\n
        https://en.wikipedia.org/wiki/Vortex_indicator\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Vortex indicator is used to identify the start of the new trend or the continuation of existing trends.\n"

        "\n Links:\n"
            "https://www.investopedia.com/terms/v/vortex-indicator-vi.asp\n"
            "https://en.wikipedia.org/wiki/Vortex_indicator\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 25):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n
            time_period(int): look back time period

        Returns:
            pandas.DataFrame: new pandas dataframe adding VI+ and VI- as two columns, preserving the columns which already exists\n
        """

        df = TR().get_value_df(df)
        df["low_prev"] = df["low"].shift(1)
        df["high_prev"] = df["high"].shift(1)

        df["VM+"] = abs(df["high"] - df["low_prev"])
        df["VM-"] = abs(df["low"] - df["high_prev"])

        df["TRn"] = df["TR"].rolling(window=time_period).sum()
        df["VM+n"] = df["VM+"].rolling(window=time_period).sum()
        df["VM-n"] = df["VM-"].rolling(window=time_period).sum()

        df["VI+"] = df["VM+n"] / df["TRn"]
        df["VI-"] = df["VM-n"] / df["TRn"]

        df = df.drop(["low_prev", "high_prev", "VM+", "VM-", "TR", "TRn", "VM+n", "VM-n"], axis=1)
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series,  time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n
            time_period(int): look back time period

        Returns:
            pandas.Series: A tuple containing vi_plus and vi_minus values
        """

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        self.df = TR().get_value_df(self.df)
        self.df["low_prev"] = self.df["low"].shift(1)
        self.df["high_prev"] = self.df["high"].shift(1)

        self.df["VM+"] = abs(self.df["high"] - self.df["low_prev"])
        self.df["VM-"] = abs(self.df["low"] - self.df["high_prev"])

        self.df["TRn"] = self.df["TR"].rolling(window=time_period).sum()
        self.df["VM+n"] = self.df["VM+"].rolling(window=time_period).sum()
        self.df["VM-n"] = self.df["VM-"].rolling(window=time_period).sum()

        vi_plus = self.df["VM+n"] / self.df["TRn"]
        vi_minus = self.df["VM-n"] / self.df["TRn"]


        self.df = pd.DataFrame(None)

        return vi_plus, vi_minus

class VO:
    """
    VO -> Volume Oscillator

    Volume Oscillator is used to identify the expansion or the contraction of volumes. \n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/VolumeOscillator.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Volume Oscillator is used to identify the expansion or the contraction of volumes. \n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/VolumeOscillator.php5\n"
        )

        return info


    def get_value_df(self, df: pd.DataFrame, short_time_period: int = 9, long_time_period: int = 26):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with volume values\n
            short_time_period(int): look back time period for short term moving average\n
            long_time_period(int): look back time period for long term moving average\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding VO as new column, preserving the columns which already exists\n
        """

        df["short_ema"] = df["volume"].ewm(span=short_ema).mean()
        df["long_ema"] = df["volume"].ewm(span=long_ema).mean()

        df["VO"] = ((df["short_ema"] - df["long_ema"]) /
                    df["long_ema"]) * 100
        df = df.drop(["short_ema", "long_ema"], axis=1)

        return df

    def get_value_list(self, volume_values: pd.Series, short_ema: int = 9, long_ema: int = 26):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            volume_values(pandas.Series): 'Volume' values\n
            short_time_period(int): look back time period for short term moving average\n
            long_time_period(int): look back time period for long term moving average\n

        Returns:
            pandas.Series: A pandas Series of VO values
        """

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
    """
    ROCV -> Rate of Change Volume

    ROCV indicator is used to identify whether the price movement is confirmed by trading volume.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/VolumeRateOfChange.php5\n
        https://www.investopedia.com/articles/technical/02/091002.asp\n
    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "ROCV indicator is used to identify whether the price movement is confirmed by trading volume.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/VolumeRateOfChange.php5\n"
            "https://www.investopedia.com/articles/technical/02/091002.asp\n"
        )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 12):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with volume values\n
            time_period(int): look back time period\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding ROCV as new column, preserving the columns which already exists\n
        """

        df["prev_volume"] = df["volume"].shift(time_period)
        df["ROCV"] = (df["volume"] - df["prev_volume"]
                      ) / df["prev_volume"] * 100
        df = df.drop(["prev_volume"], axis=1)

        return df

    def get_value_list(self, volume_values: pd.Series, time_period: int = 12):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            volume_values(pandas.Series): 'Volume' values\n
            time_period(int): look back time period\n


        Returns:
            pandas.Series: A pandas Series of ROCV values
        """
        self.df = pd.DataFrame({
            "volume": volume_values
        })
        self.df["prev_volume"] = self.df["volume"].shift(time_period)
        rocv_values = (self.df["volume"] - self.df["prev_volume"]
                       ) / self.df["prev_volume"] * 100
        self.df = pd.DataFrame(None)

        return rocv_values


class WCL:
    """
    WCL -> Weighted Close

    Weighted Close is a type of technical indicator which averages price of each period.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/WeightedClose.php5\n
    """

    def __init__(self):
        self.df = pd.DataFrame(None)

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Weighted Close is a type of technical indicator which averages price of each period.\n"

        "\n Links:\n"
            "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/WeightedClose.php5\n"
            )

        return info
    def get_value_df(self, df: pd.DataFrame):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n

        Returns:
            pandas.DataFrame: new pandas dataframe adding WCL as new column, preserving the columns which already exists\n
        """

        df["WCL"] = (df["high"] + df["low"] + (2 * df["close"])) / 4
        return df

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n

        Returns:
            pandas.Series: A pandas Series of WCL values
        """

        self.df = pd.DataFrame({
            "high": high_values,
            "low": low_values,
            "close": close_values
        })
        wcl_values = (self.df["high"] + self.df["low"] +
                      (2 * self.df["close"])) / 4

        return wcl_values


class WilliamsR:
    """
    WilliamsR -> Williams R indicator

    Williams R is tries to determine overbought and oversold levels of an asset.\n

    \n Links:
        http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/WilliamsAccumulationDistribution.php5\n
        https://www.investopedia.com/terms/w/williamsr.asp\n
        https://en.wikipedia.org/wiki/Williams_%25R\n

    """

    def __init__(self):
        self.df = pd.DataFrame()

    def info(self):
        """
        Provides basic information about the indicator
        """

        info = (
        "Williams R is tries to determine overbought and oversold levels of an asset.\n"

        "\n Links:\n"
                "http://www.ta-guru.com/Book/TechnicalAnalysis/TechnicalIndicators/WilliamsAccumulationDistribution.php5\n"
                "https://www.investopedia.com/terms/w/williamsr.asp\n"
                "https://en.wikipedia.org/wiki/Williams_%25R\n"
            )

        return info

    def get_value_df(self, df: pd.DataFrame, time_period: int = 14):
        """
        Get The expected indicator in a pandas dataframe.

        Args:
            df(pandas.DataFrame): pandas Dataframe with high, low, and close values\n
            time_period: look back time period

        Returns:
            pandas.DataFrame: new pandas dataframe adding WilliamsR as new column, preserving the columns which already exists\n
        """

        self.df["highest high"] = df["high"].rolling(
            window=time_period).max()
        self.df["lowest low"] = df["low"].rolling(
            window=time_period).min()
        df["WilliamsR"] = 100 * (df["close"] - self.df["highest high"]) / \
            (self.df["highest high"] - self.df["lowest low"])
        self.df = pd.DataFrame(None)

    def get_value_list(self, high_values: pd.Series, low_values: pd.Series, close_values: pd.Series, time_period: int = 14):
        """
        Get The expected indicator in a pandas series.\n\n
        Args:
            high_values(pandas.Series): 'High' values\n
            low_values(pandas.Series): 'Low' values\n
            close_values(pandas.Series): 'Close' values\n
            time_period: look back time period

        Returns:
            pandas.Series: A pandas Series of Williams R values
        """

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
