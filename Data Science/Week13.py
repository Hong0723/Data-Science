# 두 집단의 평균에 대한 가설검정

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df= pd.read_csv("ind_ttest.csv")
df

# 데이터 탐색 
df.head()
df.groupby('group').count() # 그룹별 표본 크기
df.groupby('group').mean() # 그룹별 평균
df.groupby('group').boxplot(grid=False)
plt.show()

group_1 = df.loc[df.group=='A', 'height']
group_2 = df.loc[df.group=='B', 'height']
group_1
group_2

# 정규성 검정
stats.shapiro(group_1)
stats.shapiro(group_2)

# 등분산성 검정
stats.levene(group_1, group_2)

# 독립표본 T-검정
result = stats.ttest_ind(group_1,group_2,equal_var=True)
result


import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('paired_ttest.csv')

# 데이터 탐색
df.head()
df[['before', 'after']].mean() # 그룹별 평균
(df['after']-df['before']).mean() # before, after 차이의 평균

fig, axes = plt.subplots(nrows=1, ncols=2 )
df['before'].plot.box(grid=False,ax=axes[0])
plt.ylim([60,100])
df['after'].plot.box(grid=False, ax= axes[1])
plt.show()

# 정규성 검정
stats.shapiro(df['after']-df['before'])

# 대응표본 T-검정
result = stats.ttest_rel(df['before'],df['after'])
result   



import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df= pd.read_csv('mw_test.csv')

# 데이터 탐색
df.head()
df.groupby('group').count() # 그룹별 표본 크기
df.groupby('group').mean() # 그룹별 평균
df.groupby('group').boxplot(grid=False)
plt.show()

group_1 = df.loc[df.group=='A','score']
group_2 = df.loc[df.group=='B','score']

# 맨 휘트니 검정
stats.mannwhitneyu(group_1,group_2)


import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('wilcoxon_test.csv')
df

# 데이터 탐색
df.mean() # 그룹별 평균
(df['post']-df['pre']).mean() # pre, post 차이의 평균

fig,axes= plt.subplots(nrows=1, ncols=2)
df['pre'].plot.box(grid=False, ax= axes[0])
plt.ylim([60,100])
df['post'].plot.box(grid=False,ax= axes[1])
plt.show()

# 월콕슨 부호순위 검정
stats.wilcoxon(df['pre'],df['post'])



from scipy import stats

men = [10,10]
women = [15,65]

# 카이제곱 검정
stats.chi2_contingency([men,women])

from scipy import stats

Group_A = [7,3]
Group_B = [2,9]

# 기대 빈도
stats.chi2_contingency([Group_A,Group_B])[3]

# 피셔 정확 검정
stats.fisher_exact([Group_A, Group_B])


