import pandas as pd
df = pd.read_csv("./sample_submission.csv")   # val icin kullandiginiz csv'nin gercek yolu
print(df["ClassId"].value_counts())
print("toplam benzersiz goruntu:", df["ImageId"].nunique())