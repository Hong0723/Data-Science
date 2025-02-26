import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset('iris')

sns.set_theme(style='whitegrid', rc={'figure.figsize':(5,5)})

sns.barplot(data = df,
             x='petal_length',
             y='species',
             estimator='mean',
             )
plt.subplots_adjust(left = 0.2)
plt.show()




data ={
'month':  ['1','2','3','4','5','6','7','8','9','10','11','12'],
'late1': ['5','8','7','9','4','6','12','13','8','6','6','4'],
'late2':  ['4','6','5','8','7','8','10','11','6','5','7','3']}
df = pd.DataFrame(data)
df