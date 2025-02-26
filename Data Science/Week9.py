############################### 결측값 ###############################
#####################################################################

# 코드 7-1
import pandas as pd
 
score = pd.Series([30,20,40,pd.NA,30,pd.NA])
score
score.sum()  # 결측값을 제외하고 계산  (합)
score.mean() # 결측값을 제외하고 계산  (평균)
score + 5  # 결측값은 산술 연산 안됨
 
pd.isna(score) # 결측값인지 여부 확인 (결측값 인지)
pd.isna(score).sum()  # 결측값의 갯수 확인

score.size # 값의 개수 (결측값 포함)
score.count() # 값의 개수 (결측값 제외)

pd.notna(score)  # 결측값이 아닌지 여부 확인 (숫자인지)
pd.notna(score).sum() # 결측값이 아닌 값의 개수 확인

score = score.dropna() # 결측값 제거
score   
score = score.reset_index(drop =True)  # 인덱스 초기화
score

# 코드 7-2
import pandas as pd
import numpy as np

# 결측값을 포함하는 데이터프레임 생성

df = pd.read_csv('iris.csv')
df.iloc[0,1] = pd.NA ; df.iloc[0,2] = pd.NA  # 특정 인덱스 NAN 값으로 변경
df.iloc[1,2] =  np.nan ; df.iloc[2,3] = None # 특정 인덱스 NAN 값으로 변경
df.head()                                        

# 결측값의 확인
df.isnull().sum() # 컬럼별 결측값 확인  (컬럼별로 결측값을 몇개씩 가지고 있는지?)
df.isnull().sum(axis =1) # 행별 결측값 확인 (행별로 결측값을 몇개씩 가지고 있는지?)      axis = 0 이라면 행을 기준으로 작업을 수행하는데 , 각 열에 대해 동작을 수행한다고 보면 됨.
df.loc[df.isnull().sum(axis=1)>0,:] # 결측값 행 출력                                 axis = 1 이라면 열을 기준으로 작업을 수행하는데, 각 행에 대해 동작을 수행한다고 보면 됨.

# 결측값의 제거
df = df.reset_index(drop = True) # 인덱스 초기화
df.head()


# 코드 7-3
# 결측값이 많을때 결측값들을 모두 제거하면 남는게 별로 없어서 분석이 어려울땐 결측값을 적당한 값으로 추정하여 대체 한 후 분석

import pandas as pd
from sklearn.impute import KNNImputer # 결측값 추정에 사용 

df_org = pd.read_csv('iris.csv')
df_miss = df_org.copy()

df_miss.iloc[0,3] = pd.NA ; df_miss.iloc[0,2] = pd.NA  # ;는 그냥 한줄에 여러 명령을 실행하려고 쓴것뿐임
df_miss.iloc[1,2] = None ; df_miss.iloc[2,3] = None
df_miss.head(4)

# 결측값 추정
tmp_np = df_miss.iloc[:,:4].to_numpy() # 넘파이 배열 변환
imputer = KNNImputer(n_neighbors=5) # 추정 모델 정의
tmp_np = imputer.fit_transform(tmp_np) # 결측값 추정
df_miss.iloc[:, :4] = tmp_np # 추정값 -> 결측값         

# 추정값의 정확도 확인
df_miss.head(4)
df_org.head(4)   # 추정값은 결국 어쩔 수 없이 오차를 포함함


# 코드 7-4
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('iris.csv')
sw  = df.Sepal_Width

# Z-score 이용
z = np.abs(stats.zscore(sw))
outliers = sw[ z>2]
print(outliers)

# IQR 이용
Q1 = sw.quantile(0.25)
Q3 = sw.quantile(0.75)
IQR = Q3 - Q1

outliers =sw[(sw < Q1 - IQR *1.5) | (sw > Q3 + IQR * 1.5)]
print(outliers)

# 특이값의 제거
clean = sw.loc[~sw.isin(outliers)]
len(clean)


############################## 정렬과 순위 ##############################
########################################################################

# 코드 7-5 

import pandas as pd

df = pd.read_csv('iris.csv')

# 데이터프레임의 정렬
# 오름차순 정렬
df_sorted = df.sort_values('Sepal_Length')
df_sorted.head(10)

# 내림차순 정렬
df_sorted = df.sort_values('Sepal_Length', ascending = False)
df_sorted.head(10)

# 여러 개의 기준 컬럼 적용
df_sorted = df.sort_values(['Specis', 'Sepal_Width'])
df_sorted.head(10)

# 코드 7-6

import pandas as pd
df = pd.read_csv('iris.csv')

# 순위
df['Petal_Length'].rank().astype(int) # 오름차순 순위
df['Petal_Length'].rank(ascending = False).astype(int) # 내림차순 순위


############################## 표본추출과 조합 ##############################
############################################################################

# 코드 7-7 
import pandas as pd
import itertools

df = pd.read_csv('iris.csv')

# 임의 표본추출
df20 = df.sample(n=20, random_state=123)
df20

# 층화 표본추출
stratified = df.groupby('Species').apply(lambda x: x.sample(frac=0.2, random_state = 123))
startified

# 조합
species = df.Species.unique()
comb = list(itertools.combinations(species,2))
comb

 ############################## 데이터 집계  ##############################
##########################################################################

# 코드 7-8
# 문제 1 
# iris 데이터셋에서 각 품종별로 꽃잎 꽃받침의 폭과 길이의 평균을 보이시오

import pandas as pd
df = pd.read_csv('iris.csv')
df_agg = df.groupby('Species').mean()
df_agg

df_agg = df.groupby('Species').std()
df_agg

 # 코드 7-9
 # 문제 2
# mtcars 데이터셋에서 cyl와 vs을 기준으로 다른 컬럼들의 최대값을 보이시오

import pandas as pd
df = pd.read_csv('mtcars.csv')
df_agg = df.groupby(['cyl', 'vs']).max()
df_agg