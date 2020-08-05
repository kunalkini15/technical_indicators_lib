# Technical Indicators
> Technical indicators library provides means to derive stock market technical indicators.



Provides multiple ways of deriving technical indicators using raw OHLCV(Open, High, Low, Close, Volume) values. 

Supports 35 technical Indicators at present.

Provides 2 ways to get the values,

1. You can send a pandas data-frame consisting of required values and you will get a new data-frame with required column appended in return.
	Note: make sure the column names are in lower case and are as follows
		- Open values should be named 'open'
		- High values should be named 'high'
		- Low values should be named 'low'
		- Close values should be named 'close'
		- Volume values should be named 'volume'


2. You can send numpy arrays or pandas series of required values and you will get a new pandas series in return. 

## Installation

```
pip install  technical_indicators_lib
```

## Usage



