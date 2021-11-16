import pandas as pd

dic_1 = {"left":3, "right":900}
dic_2 = {"湖北":0.000073, "江苏":0.009000,"上海":100}

series_1 = pd.Series(dic_1)
print(series_1.idxmax())