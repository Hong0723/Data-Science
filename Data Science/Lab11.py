import pandas as pd
from scipy import stats

df= pd.read_csv('iris.csv')
df

group_1 = df.loc[df.Species=='setosa','Petal_Length']
group_2 = df.loc[df.Species=='versicolor','Petal_Length']
    
group_1.describe()
group_2.describe()

stats.shapiro(group_1)
stats.shapiro(group_2)
stats.levene(group_1,group_2)
stats.ttest_ind(group_1,group_2,equal_var=False)


## 2번

from scipy import stats

M = [67,36]
N = [52,45]

stats.chi2_contingency([M,N])[3] 
stats.chi2_contingency([M,N])
