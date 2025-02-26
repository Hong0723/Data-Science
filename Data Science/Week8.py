import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('mtcars.csv')
df

df.plot.scatter(x='wt', y= 'mpg')
plt.show()

df.plot.scatter(x='wt',
                y='mpg',
                s=50,
                c='red',
                marker='s')
plt.show()

vars=['mpg', 'disp', 'drat', 'wt']
pd.plotting.scatter_matrix(df[vars])
plt.show()

df= pd.read_csv('iris.csv')

dict= {'setosa' : 'red', 'versicolor':'green', 'virginica' : 'blue'}

colors = list(dict[key] for key in df.Species)
colors

df.plot.scatter(x='Petal_Length',
                y='Petal_Width',
                s=30,
                c=colors,
                marker = 'o')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

beers =[5,2,9,8,3,7,3,5,3,5]
bal =[0.1, 0.03, 0.19, 0.12, 0.04, 0.0095, 0.07,0.06, 0.02,0.05]

dict = {'beers' : beers, 'bal' : bal}
df=pd.DataFrame(dict)
df

df.plot.scatter(x='beers', y= 'bal', title= 'Beers~Blood Alcohol Level')

m ,b = np.polyfit(beers,bal,1) # 회귀식 계산
plt.plot(beers,m*np.array(beers)+ b) # 회귀선 출력
plt.show()

df['beers'].corr(df['bal'])

df2 = pd.read_csv('iris.csv')
df2.columns
df2 = df2.loc[: ,~df2.columns.isin(['Species'])]
df2.columns

df2.corr()

import pandas as pd
import matplotlib.pyplot as plt

late = pd.Series([5,8,7,9,4,6,12,13,8,6,6,4], index =list(range(1,13)))

late.plot(title = 'Late student per month', xlabel = 'month', ylabel = 'frequency', linestyle = 'solid')
plt.show()

late.plot(title = 'Late student per month', xlabel= 'month', ylabel = 'frequency', linestyle = 'dashed', marker = 'o')
plt.show()