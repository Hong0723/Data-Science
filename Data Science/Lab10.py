import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import konlpy
import PIL
import numpy as np

icon = PIL.Image.open('cloud.png').convert("RGBA")
img = PIL.Image.new(mode = 'RGBA', size = icon.size, color=(255,255,255))
img.paste(icon,icon)
img = np.array(img)
df = pd.read_fwf('rating_train.txt').iloc[:,0]
kkma = konlpy.tag.Kkma()
nouns = df.apply(kkma.nouns)
nouns = nouns.explode()
df_word = pd.DataFrame({'word': nouns})
df_word['count'] = df_word['word'].str.len()
df_word = df_word.query('count >= 2')
df_word
df_word = df_word.groupby('word', as_index = False)
df_word = df_word.count().sort_values('count', ascending=False)
dic_word = df_word.set_index('word').to_dict()['count']
wc = WordCloud(random_state =123,
               font_path = 'malgun.ttf',
               width=400,
               height=400,
               background_color='white',
               mask = img,
               colormap='plasma')
img_wordcloud = wc.generate_from_frequencies(dic_word)
plt.figure(figsize = (10,10))
plt.axis('off')
plt.imshow(img_wordcloud)
plt.show()



import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

data = 'DS_10장과제_grocery_dataset.txt'
with open(data,'r') as file:
    transactions = [line.strip().split(',') for line in file.readlines()]    
te = TransactionEncoder()
trans_matrix = te.fit(transactions).transform(transactions)
df = pd.DataFrame(trans_matrix, columns=te.columns_)
freq_item = apriori(df,min_support=0.01,use_colnames=True)
num_itemsets = len(freq_item)
rules = association_rules(df=freq_item,metric='confidence',min_threshold=0.1,num_itemsets=num_itemsets)
rules.sort_values('confidence', ascending=False,inplace=True)
rules.head(10)
