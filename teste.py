import pandas as pd

n=pd.Series([-5.,-6.,-4.,-8]).shift(-1)
t=n.pct_change()

print(t)