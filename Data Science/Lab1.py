import pandas as pd

blood = pd.Series(['A','B','O','AB'])

blood.iloc[1]

df = pd.DataFrame([[85,96], [14, 12]])
df

df.iloc[1,1]

score = pd.DataFrame([[85, 96, 40, 95],
                      [73, 69, 45, 80],
                      [78, 50, 60, 90]], 
                     index = ['John', 'Jane', 'Tom'],
                     columns = ['KOR', 'ENG', 'MATH', 'SCI'])
score.index
score.columns

score.iloc[0,2]
score.columns[1]
score.loc['Jane','ENG']

print(score.iloc[1,'ENG'])
print(score.loc['Tom',2])
