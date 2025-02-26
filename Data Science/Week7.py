import pandas as pd

favorite = pd.Series(['WINTER', 'SUMMER', 'SPRING', 'SUMMER', 'SUMMER', 'FALL', 'FALL', 'SUMMER', 'SPRING', 'SPRING'])
favorite
favorite.value_counts()
favorite.value_counts()/favorite.size

fd = favorite.value_counts()
type(fd)
fd['SUMMER']
fd.iloc[2]

import matplotlib.pyplot as plt

fd.plot.bar(xlabel = 'Season',
            ylabel = 'Frequency',
            rot = 0,
            title = 'Favorite Season')

plt.subplots_adjust(left=0.2)
plt.show()

fd.plot.bar(xlabel= 'Season',
            ylabel = 'Frequency',
            rot = 0,
            title = 'Favortie Season')
plt.show()

fd.plot.barh(xlabel ='Frequency',
             ylabel = 'Season',
             rot = 0,
             title = 'Favorite Season')
plt.show()

fd.plot.pie(ylabel ='',
            autopct = '%1.0f%%',
            title = 'Favorite Season')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
colors = pd.Series([2,3,2,1,1,2,2,1,3,2,1,3,2,1,2])
fd = colors.value_counts()
fd.index
fd.index = ['red','green','blue']
fd

fd.plot.bar(xlabel='Color', ylabel = 'Frequency', rot = 0, title ='Favorite Color')
plt.show()

fd.plot.pie(ylabel='', autopct ='%1.0f%%', title = 'Favorite Color')
plt.show()

from scipy import stats

ds= [60,62,64,65,68,69]
weight = pd.Series(ds)
ds.append(120)
weight_heavy = pd.Series(ds)
weight
weight_heavy

weight.mean()
weight_heavy.mean()
weight.median()
weight_heavy.median()
stats.trim_mean(weight,0.2)
stats.trim_mean(weight_heavy,0.2)

import numpy as np

mydata = [60, 62, 64, 65 ,68 ,69 ,120]
mydata = pd.Series(mydata)
mydata.quantile(0.25)
mydata.quantile(0.5)
mydata.quantile(0.75)
mydata.quantile([0.25,0.5,0.75])

mydata.quantile(np.arange(0,1, 1.1, 0.1))
mydata.describe()

import pandas as pd
import numpy as np

mydata = pd.Series([60,62,64,65,68,69,72])
mydata.var()
mydata.std()

import matplotlib.pyplot as plt
df = pd.read_csv('cars.csv')
dist = df ['dist']
dist

dist.plot.hist()
plt.show()

dist.plot.hist(bins =6)
plt.show()

dist.value_counts(bins=6, sort = False)
dist.plot.hist(bins =6,
               title = 'Braking distance',
               xlabel = 'distance',
               ylabel = 'frequency',
               color = 'g')
plt.show()

df = pd.read_csv('cars.csv')
dist = df['dist']
dist

dist.plot.hist()
plt.show()

dist.plot.hist(bins=6)
plt.show()

dist.value_counts(bins =6, sort=False)

dist.plot.box(title = 'Breaking distance')
plt.show()

df = pd.read_scv('')
df.boxplot(column = 'Petal_Length',
           by = 'Species',
           grid = False)
plt.suptitle('')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

favorite = pd.Series(['WINER', 'SUMMER', 'SPRING', 'SUMMER', 'SUMMER', 'FALL', 'FALL', 'SUMMER', 'SPRING', 'SPRING'])
fd = favorite.value_counts()

fd.plot.bar(xlabel = 'Season',
            ylabel = 'Frequency',
            rot = 30,
            title = 'Favorite Season',
            color = 'b',
            grid = True,
            figsize = (6,4))
plt.subplots_adjust(bottom = 0.2)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv('cars.csv')
fig, axes = plt.subplots(nrows=2, ncols=2)


df['Petal_Length'].plot.hist(ax=axes[0,0])


import pandas as pd
import matplotlib.pyplot as plt

hobby = pd.Series(['등산', '낚시', '골프', '수영', '등산', '등산', '낚시', '수영', '등산', '낚시', '수영', '골프'])
hobby

fd = hobby.value_counts()
fd

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes =plt.subplots(nrows=1, ncols =2)
fd.plot.bar(ax = axes[0])
fd.plot.pie(ax=axes[1])
fig.suptitle('선호 취미 분포', fontsize =14)
plt.show()
