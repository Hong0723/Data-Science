import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('user_behavior_dataset.csv')
df
plt.scatter( df[df['Gender'] == 'Male']['App_Usage_Time'],
                df[df['Gender'] == 'Male']['Number of Apps Installed'],
                 s=30,
                 c= 'red',
                 marker = 'o',
                 label = 'Male')

plt.scatter( df[df['Gender'] == 'Female']['App_Usage_Time'],
                df[df['Gender'] == 'Female']['Number of Apps Installed'],
                 s=30,
                 c='blue',
                 marker = '^',
                label = 'Female' )

plt.xlabel('App_Usage_Time')
plt.ylabel('No_of_Apps')
plt.show()

df = df.iloc[:, 3:10]
df
df.columns = ['App_Usage_Time', 'Screen_On_Time', 'Battery_Drain', 'No_of_Apps', 'Data_Usage', 'Age', 'Gender']
df.plot.scatter( x = 'App_Usage_Time', y = 'No_of_Apps')
plt.show()  


vars = ['App_Usage_Time', 'Screen_On_Time', 'Battery_Drain', 'No_of_Apps']
pd.plotting.scatter_matrix(df[vars])
plt.show()

df['App_Usage_Time'].corr(df['No_of_Apps'])

df2= df.loc[: , ~df.columns.isin(['Gender'])]
df2.corr()

correlation_maxrix = df2.corr()
correlation_unstacked = correlation_maxrix.unstack()
sorted_correlation = correlation_unstacked.sort_values(ascending = False)
sorted_correlation.index[0],sorted_correlation[0]
sorted_correlation.index[-1], sorted_correlation[-1]

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('students.csv', thousands = ',')

df_subset = df.iloc[0:5]
df_subset[['초등학교', '중학교', '고등학교']] /= 1000000  # 학생 수를 백만 단위로 변환

 # 그래프 그리기
plt.plot(df_subset['연도'], df_subset['초등학교'], label='초등학교', marker='o', color='blue')
plt.plot(df_subset['연도'], df_subset['중학교'], label='중학교', marker='s', color='green')
plt.plot(df_subset['연도'], df_subset['고등학교'], label='고등학교', marker='^', color='red')

# 그래프 제목 및 축 레이블 설정
plt.title('연도별 초중고 학생 수 (단위: 백만 명)')
plt.xlabel('연도')
plt.ylabel('학생 수 (백만 명)')

# 범례 추가
plt.legend(loc='best')

# 그래프 출력
plt.show()
