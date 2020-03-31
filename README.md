# crypto_trading_bot
Automated cryptocurrency trading bot. 

This connects to the [Binance](https://www.binance.com) API using the [python-binance](https://python-binance.readthedocs.io/en/latest/) library. *If you would like to use this out you will need to have an account with [Binance](https://www.binance.com) and access to the API using your own API keys, which you should enter into the `binance_credentials.py` file*.

The `TradingRobot` class will connect to the Binance data kline stream, which returns open, high, low, close, and volume data for each unit of the chosen interval (e.g. `'1m'`). When initialised, it will grab all historical data for that symbol (e.g. `'BTCUSDT'`) and for that interval until the current timepoint. This allows for the generation of indicator-based decisions from the very first iteration. Analyses functions can be applied (e.g. [moving average filters](https://en.wikipedia.org/wiki/Moving_average), [relative strength index](https://en.wikipedia.org/wiki/Relative_strength_index)) between iterations. After each iteration an annoying click will happen to inform you that an iteration has occurred. To stop the running you call `bot.stop()` and to plot the data you call `bot.plot()` (note that these commands during streaming will only work in an IDE). All useful data is stored and outputted into a `.csv` file in a data directory.

### Important
This is part of a personal project to have a trading bot continuously active and making trading decisions based on technical indicators. It is purely for my own educational purposes to develop further knowledge about programming and data analysis in the context of a real-world problem, while also learning a little about trading along the way. **I do not claim that this is profitable**.
