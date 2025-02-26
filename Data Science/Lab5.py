import pandas as pd
df = pd.read_csv('user_behavior_dataset.csv')

models = df['Device_Model']
usage_time = df['App_Usage_Time']

md= df['Device_Model'].value_counts()
md

md / models.size

import matplotlib.pyplot as plt

md.plot.bar(xlabel ='Device_Model',
            ylabel = 'Frequency',
            rot = 0 ,
            title = 'md')
plt.show()

md.plot.pie(ylabel = '', autopct = '%1.0f%%',title = 'md')
plt.show()

usage_time.mean()
usage_time.median()
usage_time.std()
usage_time.quantile([0.25, 0.5, 0.75])  

usage_time.plot.hist(bins =6)
plt.show()

usage_time.plot.box(title = 'Usage_Time')
plt.show()
df.boxplot(column = 'App_Usage_Time',by= 'Operating_System', grid = False)
plt.suptitle('')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows =1, ncols= 2)
usage_time.plot.hist(ax=axes[0])
usage_time.plot.box(ax=axes[1])
plt.show()