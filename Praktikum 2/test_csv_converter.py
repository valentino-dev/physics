from olib import *
import pandas as pd


pandas_table = pd.read_csv("test.csv", sep=";")
print(pandas_table)
CSV_to_PDF(pandas_table, "output.pdf")


