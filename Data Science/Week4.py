import pandas as pd
import numpy as np


temp = pd.Series([-0.8, -0.1, 7.7, 13.8, 18.0, 22.4, 25.9, 25.3, 21.0, 14.0, 9.6, -1.4])
temp

temp.index
temp.index=['1월', '2월', '3월','4월',
            '5월','6월','7월','8월',
            '9월','10월','11월','12월']
temp

temp.size
len(temp)
temp.shape
temp.dtype

temp.iloc[2]
temp.loc['3월']
temp.iloc[[3,5,7]]
temp.loc[['4월','6월','8월']]
temp.iloc[5:8] # : 을 쓰면 ~부터 ~까지 -> 5월부터 8월
temp.loc['6월': '9월']
temp.iloc[:4] # 인덱스 0~3의 값
temp.iloc[9:] # 인덱스 9~11의 값
temp.iloc[:] # 인덱스 0~11의 값

temp.loc[temp >= 15]
temp.loc[(temp >= 15) & (temp < 25) ]
temp.loc[ (temp < 5) | (temp >= 25)]
march = temp.loc['3월']
temp.loc[temp < march]
temp.where(temp >= 15)
temp.where(temp >= 15).dropna()

temp.loc[temp >= 15]
temp.loc[temp >= 15].index
temp + 1
2 * temp + 0.1
temp + temp
temp.loc[temp >= 15] + 1 

temp.sum()
temp.mean()
temp.median()
temp.max()
temp.min()
temp.std()
temp.var()
temp.abs()
temp.describe()

salary = pd.Series([20, 15, 18, 30])
score = pd.Series([75, 80, 90, 60], 
                  index = ['KOR', 'ENG', 'MATH', 'SOC'])
salary
score

# 값의 변경
score.iloc[0] = 85
score
score.loc['SOC'] = 65
score
score.loc[['ENG', 'MATH']] = [70,80]
score

# 값의 추가 ( 레이블의 인덱스가 있는 경우)
score.loc['PHY'] = 50
score
score.iloc[5]= 90 # 에러발생 why? 인덱스가 3까지 밖에 없어서 5로는 못감

# 값의 추가 ( 레이블의 인덱스가 없는 경우)
next_idx = salary.size
salary.iloc[next_idx] = 33 # 인덱스가 없기에 에러 발생
salary.loc[next_idx] = 33
salary

# _append() 메소드를 이용한 값의 추가
new = pd.Series({'MUS' : 95})
score._append(new) #  score 점수 변경 없음
score
score = score._append(new) # score 점수 변경
score

salary._append(pd.Series([66]), ignore_index = True)
salary = salary._append(pd.Series([66]), ignore_index = True)   
salary

score.drop('PHY')
score
score = score.drop('PHY')
score

salary = salary.drop(1)
salary

score_1 = pd.Series([80, 75, 90],
                    index = [ 'KOR', 'ENG', 'MATH'])

score_2 = score_1
score_2.loc['KOR'] = 95
score_2
score_1

score_3 = score_1.copy()
score_3.loc['KOR'] = 70
score_3
score_1