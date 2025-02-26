import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import konlpy



# (1) 데이터 준비
df = pd.read_csv('movie1_review.csv')
df
# (2) 형태소 분석기 정의
kkma = konlpy.tag.Kkma() 

# (3) 단어 데이터 프레임 만들기
nouns = df['Review'].apply(kkma.nouns) # 명사추출
nouns

nouns = nouns.explode()
nouns

# (4) 전처리 실시

# 모시 -> 티모시, imax -> 아이맥스, ...
nouns[nouns=='모시'] = '티모시'
nouns[nouns=='IMAX'] = '아이맥스'
nouns[nouns=='파트3'] = '3편'

# 글자수 2개 이상인 단어만 추출
df_word = pd.DataFrame({'word' : nouns})
df_word['count'] = df_word['word'].str.len()
df_word =df_word.query('count >= 2')
df_word 

# 단어 빈도수 집계 및 정렬
df_word = df_word.groupby('word', as_index=False)
df_word = df_word.count().sort_values('count', ascending= False)
df_word

# 불필요한 단어 제거
del_idx = df_word.loc[df_word.word.isin(['영화', '편이', '영화관','파트'])].index(value)
df_word = df_word.drop(index=del_idx)
df_word

# (5) 워드클라우드 만들기

# 빈도수 상위 10개 단어
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df_top10 = df_word.iloc[:10, :].sort_values('count', ascending= True)
df_top10.plot.barh(x='word', y='count')
plt.show()

# 워드 클라우드
dic_word = df_word.set_index('word').to_dict()['count']
dic_word

wc = WordCloud(random_state=123,
               font_path='malgun.ttf',
               width=400,
               height=400,
               background_color='white')

img_wordcloud = wc.generate_from_frequencies(dic_word) # 워드클라우드를 이미지로

plt.figure(figsize=(10,10)) # 크기 지정하기
plt.axis('off') # 축 없애기
plt.imshow(img_wordcloud) # 결과 보여주기
plt.show()


######### 구매 패턴 분석 ##########

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('chipotle.csv')
df
df.info()
df.head()


# (2) 데이터 탐색

len(df['Item'].unique()) # 음식의 종류
temp = df[df.item_price ==df.item_price.max()] # 가장 비싼 음식
temp = temp[['Item','item_price']].drop_duplicates()
temp

temp = df[df.item_price==df.item_price.min()] # 가장 저렴한 음식
temp = temp[['Item','item_price']].drop_duplicates()
temp

len(df['Transaction'].unique())  # 트랜잭션 수 

# 많이 판매된 음식
sales_quntity = df.groupby('Item').count()
sales_quntity = sales_quntity.sort_values('Transaction', ascending = False)
sales_quntity['Transaction']

# 매출상위 10개 상품
top_ten = sales_quntity.sort_values('Transaction').tail(10)
top_ten = top_ten['Transaction']

top_ten.plot.barh(xlabel ='Transaction',
                  ylabel ='',
                  title ='Top 10 Items',
                  figsize=(9,5))

plt.subplots_adjust(left=0.2) 
plt.show()

## (3) 연관분석

# 전처리
temp = df[['Transaction','Item']].drop_duplicates()
temp = temp.groupby('Transaction')['Item'].apply(list)
temp

te = TransactionEncoder()
trans_matrix = te.fit(temp).transform(temp)
trans_matrix

basket = pd.DataFrame(trans_matrix, columns=te.columns_)
basket.head()

# 연관 규칙 검색
freq_item = apriori(df=basket,min_support= 0.01, use_colnames = True)
freq_item

rules = association_rules(df=freq_item, metric = 'lift', min_threshold = 1) 
rules.sort_values('confidence', ascending= False,inplace=True)  
rules.head(10)
rules.iloc[0,:].transpose()