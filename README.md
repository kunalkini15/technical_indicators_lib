# Technical Indicators
> Technical indicators library provides means to derive stock market technical indicators.



Provides multiple ways of deriving technical indicators using raw OHLCV(Open, High, Low, Close, Volume) values. 

Supports 35 technical Indicators at present.

Provides 2 ways to get the values,

1. You can send a pandas data-frame consisting of required values and you will get a new data-frame with required column appended in return.

	Note: make sure the column names are in lower case and are as follows,

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
```
# import dependencies
import pandas as pd
import numpy as np

# importing an indicator class
from technical_indicators_lib import OBV

# instantiate the class
obv = OBV()

# load data into a dataframe df


# Method 1: get the data by sending a dataframe
df = obv.get_value_df(df)


# Method 2: get the data by sending series values

obv_values = obv.get_value_list(df["close"], df["volume])

```



## Development

Want to contribute?

Great. Follow these steps, 

```
git clone https://github.com/kunalkini015/technical-indicators.git

cd technical_indicator_lib

pip install -r requirements.txt

```

then you are good to go. You can create a pull request or write to me at kunalkini15@gmail.com

## Todo

- Divide indicators into separate modules, such as trend, momentum, volatility, volume, etc.

- add tests.

- Add more indicators.

## Credits

Developed by Kunal Kini K, a software engineer by profession and passion. 

If you have any comments, feedbacks or queries, write to me at kunalkini15@gmail.com


