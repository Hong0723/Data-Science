import pandas as pd

################################### 2장 ######################################### 
age = pd.Series([25,34,19,45,60])
age
type(age)

data = ['Spring', 'summer', 'fall', 'Winter']
seasom = pd.Series(data)
seasom

seasom.iloc[2]

score = pd.DataFrame([[85,96,40,95],
                      [73,69,45,80],
                      [78,50,60,90]])

score
type(score)
score.index
score.columns

score.iloc[1,2]

import pandas as pd
import numpy as np

# 넘파이 1차원 배열을 판다스로 변환
w_np = np.array([65.4, 71.3, np.nan, 57.8]) # 넘파이 1차원 배열
weight = pd.Series(w_np) # 넘파이 배열을 판다스 시리즈로
weight #판다스 시리즈

#판다스 시리즈를 넘파이로 변환
w_np2 = pd.Index.to_numpy(weight)
w_np2

# 넘파이 2차원 배열로 부터 데이터 프레임생성

s_np = np.array([[85,95,40,30],
                 [54,34,57,76],
                 [65,30,10,59]])
s_np # 넘파이 2차원 배열

s_np2 = pd.DataFrame(s_np) # 넘파이 2차원 배열을 데이터프레임으로 변경
s_np2 # 판다스 데이터 프레임

# 데이터 프레임을 넘파이 2차원 배열로 변환
score_np = s_np2.to_numpy() # 데이터 프레임을 넘파이 배열로
score_np # 넘파이 2차원 배열

# 시리즈에 레이블 부여
age = pd.Series([25,45,54,34])
age  # 레이블 부여 전
age.index = ['h', 'q', 'w', 'e'] # 레이블 부여
age #레이블 부여 후
age.loc['h'] #레이블에 의한 인덱싱
age.iloc[2] # 절대위치에 의한 인덱싱

score = pd.DataFrame([[65, 45, 30, 50],
                      [60,70,80,90],
                      [60,50,49,50]])

score
score.index = ['Hong', 'Lee', 'Park']
score
score.columns = ['Eng', 'Math', 'Sci', 'Kor']
score
score.iloc[2,3]
score.loc['Hong']

age = pd.Series([25,34,19,45,60])
age.index = ['H', 'P', 'R', 'E', 'P'] # 인덱스 이름 중복 가능
age

age.iloc[3]
age.loc['P'] 

population = pd.Series([123,342,423,253,264])   # 인덱스를 문자열 말고도 숫자로도 사용 가능능
population
population.index = [10,20,30,40,50]
population
population.iloc[2]
population.loc[20]



temperature = pd.DataFrame([[-0.1, 0.0, -0.1, -0.2], # 2장 연습문제
                         [1.8,2.0,1.6,1.6],
                         [6.4,6.8,5.8,5.9],
                         [12.3,12.9,11.5,11.5],
                         [17.9,18.5,17.1,17.1],
                         [22.2,22.8,21.6,21.5]])
temperature
temperature.index = ['1월','2월','3월','4월','5월','6월']
temperature.columns = ['전북','전주','군산','부안']
temperature
temperature.iloc[2,1]
temperature.iloc[3,3]
temperature.loc['1월','군산']
temperature.loc['6월','전북']




############################# 3장 ########################################  
############################# 3장 ######################################## 

import pandas as pd
import numpy as np

temp = pd.Series([-0.8,-0.1,7.7,13.8,18.0,22.4,25.9,25.3,21.0,14.0,9.6,-1.4]) 
temp # temp 내용 확인
temp.index= ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'] #월 이름을 인덱스로 지정
temp


temp.size # 배열의 크기 ( 값의 개수)
len(temp) # 배열의 크기 ( 값의 개수)
temp.shape # 배열의 형태 ( 행의 개수가 12개고 1차원 배열이라 열에 대한 정보 생략 그러므로(12,)로 출력 됨)
temp.dtype # 배열의 자료형

# 인덱싱 : 시리즈에 저장된 값의 위치를 가리키는 체계를 의미
# 슬라이싱 : 인덱스와 조건문을 이용하여 시리즈에 저장된 값들의 일부를 잘라내어 추출하는 연산
# 시리즈 학습에 있어서 가장 핵심

temp.iloc[2] 
temp.loc['3월']
temp.iloc[[3,5,7]]  ## 인덱스 3,5,7 의 값 출력
temp.loc[['3월','5월','7월']]

temp.iloc[5:8] # 인덱스 5~7 의 값 출력
temp.loc['6월':'9월'] # 인덱스 6월~ 9월의 값 출력
temp.iloc[:4] # 인덱스 0~3의 값 출력
temp.iloc[9:] # 인덱스 9~12의 값 출력
temp.iloc[:] # 인덱스 0~12의 값 출력

# 절대위치 인덱스의 경우 start 지점 포함, but end 지점은 포함하지 않지만 
# 레이블 인덱스의 경우 start, end 지점 두개 다 모두 포함
# start 생략 : end 이전의 모든 인덱스를 의미
# end 생략 : start 부터 이후의 모든 인덱스를 의미
# start, end 모두 생략 : 전체 인덱스를 의미


# 조건문을 이용한 슬라이싱

# 조건문을 이용해 슬라이싱 하는경우 loc[] 메소드 이용
temp
temp.loc[temp >= 15]
temp.loc[(temp >= 15) & (temp < 25)] # 각 조건은 반드시 () 로 묶어준다
temp.loc[(temp < 5) | (temp >= 25)] # 각 조건은 &(and) 나 | (or) 로 연결
march = temp.loc['3월']
temp.loc[temp<march]

temp.where(temp>=15) # where() 메소드는 조건문에 의해 값을 검색 => 조건에 맞지 않는 값은 NAN으로 표시 
temp.where(temp>=15).dropna() # where() 메소드 뒤에 .dropna() 를 붙여주면 NAN 제거

# 시리즈 객ㅊ에에서 값의 위치 찾기
temp.loc[temp >= 15]
temp.loc[temp >= 15].index # 조건에 맞는 값들의 인덱스 추출

# 시리즈 객체에 대한 산술 연산
temp
temp + 1
2* temp + 0.1
temp + temp
temp.loc[temp >= 15] + 1 # 기온이 15도 넘는 월들에 대해서만 1도 추가


############ 시리즈의 통계 관련 메소드 ##################

# sum() ----- 합계 
# mean() ----- 평균
# median() ----- 중앙값
# max() ----- 최대값
# min() ----- 최소값
# std() ----- 표준값
# var() ----- 분산
# abs() ----- 절대값

temp
temp.sum()
temp.mean()
temp.median()
temp.max()
temp.min()
temp.std()
temp.var()
temp.abs()
temp.describe() # 기초 통계 정보?  => count, mean, std, min, 25%(1사분위수), 50%(2사분위수(중앙값)), 75%(3사분위 수), max 가 표시


# 시리즈 객체에 대한 값의 수정/추가/삭제

import pandas as pd
import numpy as np

salary = pd.Series([20,15,18,30]) # 레이블 인덱스가 없는 시리즈 객체
score = pd.Series([75,80,90,60],
                  index =['KOR','ENG','MATH','SOC']) # 레이블 인덱스 이름 바로 달아주기 가능
salary 
score

# 값의 변경
score.iloc[0] = 85 # 인덱스 0의 값을 변경
score
score.loc['SOC'] = 65 # 인덱스 'SOC'의 값을 변경
score_np    
score.loc[['ENG','MATH']] = [70,80] # 인덱스 'ENG','MATH'의 값을 변경
score

# 변경 하고자 하는 값을 지정할때 숫자 인덱스와 레이블 인덱스 모두 사용 가능 
# 여러 개의 값을 동시에 변경하는 경우는 변경할 값들의 인덱스를 리스트로 묶어서 입력, 인덱스 숫자만큼의 변경값들을 리스트로 묶어서할당


# 값의 추가 (레이블 인덱스가 있는 경우)
score.loc['PHY'] = 50 # 없는 인덱스 추가 
score
score.iloc[5] = 90 # 에러발생

# 값의 추가( 레이블 인덱스가 없는 경우)               ########### iloc[] 를 통해서는 배열의 크기를 늘릴 수 없음
next_idx = salary.size                              ############ loc[] 를 통해서 배열의 크기를 늘림
salary.iloc[next_idx] = 33 # 에러 발생
salary.loc[next_idx] = 33 # 정상 수행
salary

# _append() 메소드를 이용한 값의 추가

new = pd.Series({'MUS': 95})
score._append(new) # score 자체에는 변경 없음
score
score = score._append(new) # 이러면 score에 추가됨
score 

salary
salary._append(pd.Series([66]), ignore_index= True)   # 새로운 값을 시리즈 객체의 마지막 부분에 추가하는 작업은 append() 로 가능
salary = salary._append(pd.Series([66]), ignore_index = True) # 하지만 append() 메소드의 입력값은 판다스 시리즈 객체여야함
salary  # 레이블을 지정하지 않은 시리즈 객체에 대해 새로운 값을 추가할때 ignore_index = True 를 설정해줘야함


# 값의 삭제
score
score.drop('PHY') # 레이블 인덱스가 있는 경우
score # score 의 내용 변동 없음
score = score.drop('PHY')  # scre 의 내용 변동
score

salary = salary.drop(1) # 레이블 인덱스가 없는 경우
salary

 # 시리즈 객체의 복사
 
import pandas as pd
 
score_1 = pd.Series([80,75,90],
                     index = ['KOR','ENG', 'MATH'])
score_2 = score_1 # score_2 와 score_1는 동일한 객체
score_2.loc['KOR'] = 95
score_2
score_1

score_3 = score_1.copy()  # 1의 값은 건드리지 않고 3의 값을 건드리고 싶을땐 .copy() 를 사용한다.  1과 3은 독립된 객체
score_3.loc['KOR'] =70
score_3
score_1



############################# 4장 ########################################  
############################# 4장 ######################################## 

import pandas as pd

df = pd.read_csv('iris.csv') # csv 내용 읽기
df
# 데이터프레임 개체의 정보 확인
df.info() # 배열의 정보
df.shape # 배열의 형태
df.shape[0] # 행의 개수
df.shape[1] # 열의 개수

df.dtypes # 각 열의 자료형
df['Species'] = df['Species'].astype('category')
df.dtypes

df.columns # 열의 이름 보기
df.head()  # 데이터 앞부분 일부만 보기
df.tail() # 데이터 뒷부분 일부만 보기
df['Species'].unique()  # 품종 정보의 확인

df.iloc[2,3]
df.loc[3,'Petal_Width']
df.loc[[0,2,4],['Petal_Length', 'Petal_Width']] # 0행,2행,4행의 petal length 와 petalwidth 값 출력
df.loc[5:8,'Petal_Length']
df.iloc[:5,:4] # 0~4행, 0~3열의 값 출력   
df.loc[:,'Petal_Length'] # petal_length 열의 모든 행의 값 
df['Petal_Length'] # petal_length 열의 모든 값
df.Petal_Length # petal_length 열의 모든 값
df.iloc[:5,:] # 0~4행의 모든 열의 값
df.iloc[:5] # 0~4행의 모든 열의 값

# 조건문을 이용한 슬라이싱
df.loc[df.Petal_Length >= 6.5, : ] 
df.loc[df.Petal_Length >= 6.5 ]  # 모든 열의 경우 열 인덱스 생략

df.loc[df.Petal_Length >= 6.5].index

df.loc[(df.Petal_Length >= 3.5) & (df.Petal_Length <= 3.8)]
df.loc[(df.Petal_Length < 1.3) | (df.Petal_Length >6.5), ['Petal_Length','Petal_Width']]

df.where(df.Petal_Length >= 6.5).dropna()

df.loc[ : , df.columns != 'Species'] # Species 열 제외
df.loc[ : , ~df.columns.isin(['Species']) ] # Species 열 제외

df.loc[ : , df.columns != 'Species'] + 10 # 모든 값을 10 더함
df['Sepal_Length'] + df['Petal_Length'] 

df.loc[:,(df.columns != 'Sepal_Length') & (df.columns != 'Petal_Length')] 
df.loc[:, ~df.columns.isin(['Sepal_Length', 'Petal_Length'])]

df2 = df.loc[:, df.columns != 'Species']
df2.sum()  # 동일식은 df2.sum(axis=0) axis 란 축의 방향을 의미하며 axis = 0 세로방향, axis = 1 가로방향을 뜻함
df2.mean()
df2.median()
df2.max()
df2.min()
df2.std() # 표준편차
df2.var() # 분산
df2.abs() # 절댓값
df.describe() # 기초통계정보

# 데이터프레임 객채에 있는 값의 수정
df3 = df.copy()
df3.iloc[1,2] = 5.5 #1행 2열의 값 수정
df3.loc[1,'Petal_Length'] = 1.1 

df3.Petal_Length.to_list() # 시리즈를 리스트로
df3.loc[df.Petal_Length > 6.5, 'Petal_Length'] *= 100

# 데이터프레임 객체에 대한 행과 열의 추가

# 뒷부분에 새로운 행 추가
df3.shape
new_idx = df3.shape[0] # 추가할 행번호 결정
df3.loc[new_idx] = [1.1, 4.5, 3.4, 2.2 ,'setosa']
df3.tail()

# 중간에 새로운 행 추가
new_row = pd.DataFrame([[1.1,2.2,3.3,4.4,'virginica']],
                       columns=df3.columns)
new_row
df3 = pd.concat([df3.iloc[:10], new_row, df3.iloc[10:]], ignore_index= True)
df3.iloc[8:13,:]

# 여러 행의 추가
ext = pd.DataFrame([[1.2,3.5,4.3,3.1, 'setosa'],
                    [2.1,3.2,2.3,5.2,'versicolor']],
                   columns = df3.columns)\
ext
df3 = df3._append(ext, ignore_index = True)
df3

# 뒤쪽에 열 추가
new_col = df3.Petal_Length * 10
df3['new col'] = new_col
df3

# 중간에 열 추가
df4 = df.copy()
df4.insert(loc=2,column='new_col2', value= new_col)  # loc = 새로운 컬럼을 추가할 위치, column : 추가할 컬럼의 이름, value: 추가할 컬럼 데이터(시리즈)
df4

# 데이터프레임 객체에 대한 행과 열의 삭제
df5 = df.copy()

#행의 삭제
df5 = df5.drop(index= [1,3])
df5
df5 = df5.reset_index() # 인덱스 삭제한거 복구
df5

# 열의 삭제
df5 = df5.drop(columns = 'Petal_Length')
df5

# 연속된 행의 삭제
df5 = df5.drop(index= range(0,4))

df5 = df5.reset_index()
df5 = df5.drop(columns = ['Sepal_Length', 'Petal_Width']) # 삭제할 열이 여러개면 리스트로 묶어서 기술
df5



############################# 5장 ########################################  
############################# 5장 ######################################## 

# 단일 변수 범주형 데이터
import pandas as pd

favorite = pd.Series(['WINTER', 'SUMMER', 'SPRING', 'SUMMER', 'SUMMER', 'FALL','FALL', 'SUMMER', 'SPRING', 'SPRING'])
favorite    
favorite.value_counts() # 도수분표 계산
favorite.value_counts() / favorite.size # 비율 계산

fd = favorite.value_counts() # 도수분포를 fd에 저장
type(fd) # fd의 자료구조 확인
fd['SUMMER'] # SUMMER의 빈도 확인
fd.iloc[2] # 인덱스 2번째의 빈도 확인

#### 막대 그래프의 작성 ####

import matplotlib.pyplot as plt

fd.plot.bar(xlabel='Seanson',  # x축 레이블
            ylabel ='Frequency', # y축 레이블
            rot = 0, # x축 값의 회전 각도
            title ='Favorite Season') # 그래프 제목
plt.show()

# 막대그래프의 세로 출력
fd.plot.barh(xlabel = 'Frequency', # x축 레이블
             ylabel = 'Season', # y축 레이블
             rot = 0, # x축 값의 회전 각도
             title = 'Favorite Season')
plt.subplots_adjust(left = 0.2) # 그래프 왼쪽 여백
plt.show()

# 원 그래프의 작성

fd.plot.pie(ylabel = '', # y축 레이블
            autopct = '%1.0f%%', # '%1.0f%%' 는 소수점 이하 자리수는 표시 x , '%1.2f%%'는 소수점 이하 2자리수 까지 표시
            title = 'Favorite Season') 
plt.show()
 

# 숫자로 표현된 범주형 데이터
import pandas as pd
import matplotlib.pyplot as plt

colors = pd.Series([2,3,2,1,1,2,2,1,3,2,1,3,2,1,2])

fd = colors.value_counts() # 도수분포를 fd에 저장
fd
fd.index # fd의 인덱스 출력
fd.index = ['red','green','blue'] # 숫자 인덱스를 문자 인덱스로 변경
fd

# 막대그래프 출력
fd.plot.bar(xlabel = 'Colors',
            ylabel = 'Frequency',
            rot = 0,
            title = 'Favorite Colors')
plt.show()

# 원 그래프 출력
fd.plot.pie(ylabel='',
            autopct ='%1.0f%%',
            title ='Favotite Color')
plt.show()

###### 단일변수 연속형 데이터의 탐색 #######

# 중앙값 : 데이터의 값들을 크기순으로 일렬로 세웠을때 가장 중앙에 위치하는값 (짝수개라면 중앙값 2개의 평균)
# 절사평균 : 데이터의 관측값들중에서 작은 값들 하위 n%와 큰값들 상위n%를 제외하고 중간에 있는 값들 가지고 평균을 계산하는 방식

import pandas as pd
from scipy import stats


ds = [60,62,64,65,68,69]
weight = pd.Series(ds)
ds.append(120) # ds에 120 추가
weight_heavy = pd.Series(ds)
weight
weight_heavy

weight.mean() # 평균
weight_heavy.mean() # 평균

weight.median() # 중앙값
weight_heavy.median() # 중앙값

stats.trim_mean(weight,0.2)  # 절사평균 (상하위 20% 제외)
stats.trim_mean(weight_heavy,0.2) # 절사평균 (상하위 20% 제외)

### 사분위수 ###
# 주어진 데이터에 있는 값들을 크기순으로 나열했을때 이것을 4등분 하는 지점에 있는 값들을 의미
# 제1사분위수, 제2사분위수(중앙값), 제3사분위수

import pandas as pd
import numpy as np

mydata = [60,62,64,65,68,69,120]
mydata = pd.Series(mydata)
mydata.quantile(0.25) # 제1사분위수(Q1)
mydata.quantile(0.5) # 제2사분위수(Q2)
mydata.quantile(0.75) # 제3사분위수(Q3)
mydata.quantile([0.25,0.5,0.75])

mydata.quantile(np.arange(0,1,0.1)) # 10%단위로 구간을 나누어 계산
mydata.describe() # 사분위수 포함 요약 정보

#### 산포 ####
# 주어진 데이터에 있는 값들이 퍼져있는 정도

mydata = pd.Series([60,62,64,65,68,69,72])
mydata.var() # 분산
mydata.std() # 표준편차

# 히스토그램 #
# 외관상 막대그래프와 비슷한 그래프

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cars.csv')
dist = df['dist'] #제동거리
dist

# 구간 개수를 지정하여 히스토그램 그리기
dist.plot.hist() # 기본 그래프
plt.show()

dist.plot.hist(bins=6) # 막대 개수 지정
plt.show()

# 구간별 빈도구 계산
dist.value_counts(bins =6, sort = False) # sort란 메소드의 실행결과를 빈도수를 기준으로 내림차순으로 정렬할것인지 정한다
                                                # 기본값은 True 인데 False면 올림차순이다
dist.plot.hist(bins=6, # 막대 개수 지정
               title = 'Breaking distance', # 그래프 제목
               xlabel = 'distance', # x축 레이블
               ylabel = 'frequency', # y축 레이블
               color = 'g')  # 막대 색
plt.show()

dist.plot.box(title = 'Breaking distance')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

# 상자그림 그리기
df.boxplot(column = 'Petal_Length', # 상자그림 대상 열
           by = 'Species', # 그룹정보 열
           grid = False) # 격자 표시 제거
plt.suptitle('') #기본 표시 제목 제거
plt.show()

favorite = pd.Series(['WINTER','SUMMER','SPRING','SUMMER','SUMMER','FALL','FALL','SUMMER','SPRING','SPRING'])
fd = favorite.value_counts() # 도수분포를 fd에 저장

fd.plot.bar(xlabel = 'Season',
            ylabel = 'Frequency',
            rot = 30,
            title = 'Favorite Season',
            color = 'b',
            grid = True,
            figsize = (6,4))
plt.subplots_adjust(bottom = 0.2)
plt.show()

#### 그래프에 한글 표시하기 ####
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 변경
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

ser = pd.Series([1,2,3,4])
ser.plot.hist(title = ' 그래프 예제 - 히스토그램')
plt.show()


#### 한 화면에 여러개의 그래프 출력하기 ####

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
# 화면 분할 정의
fig, axes = plt.subplots(nrows=2, ncols=2)
# 각 분할 영역에 그래프 작성하기
df['Petal_Length'].plot.hist(ax= axes[0,0])
df['Petal_Length'].plot.box(ax=axes[0,1])

fd= df['Species'].value_counts()
fd.plot.pie(ax=axes[1,0])
fd.plot.barh(ax=axes[1,1])

#통합 그래프에 제목 지정
fig.suptitle('Multiple Graph Example', fontsize = 14)
plt.show()

## 실전 분석 ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
hobby = pd.Series(['등산', '낚시','골프','수영','등산','등산','낚시','수영','등산','낚시','수영','골프'])
fd = hobby.value_counts()
fd
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

fig,axes = plt.subplots(nrows=1, ncols=2)
fd.plot.bar(ax=axes[0])
fd.plot.pie(ax=axes[1])
fig.suptitle('선호 취미 분포',  fontsize= 14)
plt.show()

df = pd.read_csv('BostonHousing.csv')
house_price = df['medv']
house_price.mean()
house_price.median()
house_price.quantile([0.25,0.5,0.75])


############################# 6장 ########################################  
############################# 6장 ######################################## 

# 산점도 : 2개의 변수로 구성된 데이터의 분포를 알아보는 그래프

df = pd.read_csv('mtcars.csv')
df

# 기본 산점도
df.plot.scatter(x='wt', y='mpg')
plt.show()

# 매개변수 조정 산점도
df.plot.scatter(x='wt', #x축
                y='mpg', #y축
                s=50, # 점의 크기
                c='red', # 점의 색깔
                marker='s') # 점의 모양
plt.show()

# 다중 산점도의 작성
vars=['mpg','disp','drat','wt']
pd.plotting.scatter_matrix(df[vars])
plt.show()

# 그룹이 있는 다중 산점도의 작성
df = pd.read_csv('iris.csv')
dict = {'setosa' : 'red', 'versicolor': 'green', 'virginica' : 'blue'}
colors = list(dict[key] for key in df.Species) # 각 점의 색을 지정
colors 

df.plot.scatter(x='Petal_Length',
                y='Petal_Width',
                s= 30,
                c = colors,
                marker = 'o')
plt.show()

## 상관 분석 ##
# 어느정도나 선형성을 보이는지 수치상으로 나타낼 수 있는 방법

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


beers = [5,2,9,8,3,7,3,5,3,5] # 음주량
bal = [0.1,0.03,0.19,0.12,0.04,0.0095,0.07, 0.06, 0.02,0.05] # 혈중알콜농도

dict = {'beers' :beers, 'bal' : bal} # 딕셔너리 정의
df = pd.DataFrame(dict) # 데이터프레임 생성
df
df.plot.scatter(x='beers', y= 'bal', title = 'Beers~Blood Alcohol Level') # 산점도
plt.show()

# 회귀식 계산
m,b= np.polyfit(beers,bal,1)
# 회귀선 출력
plt.plot(beers,m * np.array(beers) + b)

plt.show()

# 두 변수간 상관계수 계산
df['beers'].corr(df['bal']) # 상관계수 계산

# 여러 변수들간 상관계수 
df2=pd.read_csv('iris.csv')
df2.columns
df2= df2.loc[:, df2.columns != 'Species'] # 품종 열 제외
df2.columns

df2.corr() # 상관계수 계산

# 시계열 데이터
# 시간 변화에 따라 데이터를 수집한 경우

late = pd.Series([5,8,7,9,4,6,12,13,8,6,6,4],
                 index = list(range(1,13)))
late.plot(title = 'Late student per month',
          xlabel = 'month',
          ylabel = 'frequency',
          linestyle = 'solid')
plt.show()
late.plot(title = 'Late student per month',
          xlabel = 'month',
          ylabel = 'frequency',
          linestyle = 'dashed',
          marker = 'o')
plt.show()