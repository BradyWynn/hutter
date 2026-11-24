import matplotlib.pyplot as plt
import polars as pl

df = pl.read_csv("log.csv")

plt.plot(df["bytes"][5:])
plt.plot(df["bytes"][5:].ewm_mean(com=3))
plt.grid()
plt.show()
