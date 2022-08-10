import pandas as pd
print("he lekker ding")

df = pd.DataFrame(["he", "lekker", "ding"])

df.to_csv("/export/home/embakker/test.csv", index=False, sep='|')