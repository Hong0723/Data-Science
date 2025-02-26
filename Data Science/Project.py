import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
df = pd.read_csv('한국전력거래소_시간별 전국 전력수요량_20231231.csv', encoding='euc-kr')
df['날짜'] = pd.to_datetime(df['날짜'])

#  1. 시간별 1년동안의 평균 수요량
time_columns=[str(i) + "시" for i in range(1,25)]
fd = df[time_columns].mean()
fd.plot.line(xlabel="Time",
            ylabel="평균 전력 소요량",
            title="시간별 1년간의 평균 소요량")
plt.subplots_adjust(bottom=0.2)  
plt.show()

# 2. 월별 전력수요량 총합 비교
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
df['월'] = df['날짜'].dt.month
time_columns = [f"{i}시" for i in range(1,25)]
df['월별_합계'] = df[time_columns].sum(axis=1)
monthly_total= df.groupby('월')['월별_합계'].sum()
monthly_total.plot.bar(xlabel='월',
                       ylabel='전력소모량 총합',
                       title='월별 전력소요량 총합')
plt.subplots_adjust(bottom=0.2)  
plt.show()

# 3. 계절별 평균 전력 수요량
def get_season(month) :
    if month in [12,1,2] : return '겨울'
    elif month in [3,4,5] : return '봄'
    elif month in [6,7,8] : return '여름'
    elif month in [9,10,11] : return '가을'
df['계절'] = df['날짜'].dt.month.apply(get_season)
df['총수요량'] = df.iloc[:,1:25].sum(axis=1)
seasonal_demand = df.groupby('계절')['총수요량'].sum().reset_index()   
plt.pie(
    seasonal_demand['총수요량'],             
    labels=seasonal_demand['계절'],         
    autopct='%1.1f%%',                           
    startangle=90,                               
    colors=sns.color_palette('coolwarm', 4),    
    wedgeprops={'edgecolor': 'black'},          
    textprops={'fontsize': 12}                  
)
plt.show()

# 4. 요일별 전력 소요량
df['요일'] = df['날짜'].dt.day_name(locale='ko_KR') # 한글 요일로 표시
df['총수요량']= df.iloc[:,1:25].sum(axis=1)
weekday_demand=df.groupby('요일')['총수요량'].mean().reset_index()
weekday_order=['월요일','화요일','수요일','목요일','금요일','토요일','일요일']
weekday_demand['요일']= pd.Categorical(weekday_demand['요일'], categories=weekday_order,ordered=True)
weekday_demand=weekday_demand.sort_values('요일')
plt.figure(figsize=(10, 6))
sns.boxplot(x='요일', y='총수요량', data=df, order=weekday_order, palette='coolwarm')
plt.title('요일별 전력수요량 분포', fontsize=16)
plt.xlabel('요일', fontsize=12)
plt.ylabel('총 전력수요량 (MW)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 5. 요일 및 시간대별 
df['요일'] = df['날짜'].dt.day_name(locale='ko_KR')  
time_columns = [f"{i}시" for i in range(1, 25)]  
hourly_demand = df.groupby('요일')[time_columns].mean()
weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
hourly_demand = hourly_demand.reindex(weekday_order)  
plt.figure(figsize=(12, 8))  
sns.heatmap(
    hourly_demand,          
    cmap='coolwarm',       
    linewidths=0.5,         
    linecolor='gray',      
    cbar_kws={'label': '평균 전력소요량 (MW)'} 
)
plt.title('요일 및 시간대별 평균 전력소요량', fontsize=16)
plt.xlabel('시간대', fontsize=12)
plt.ylabel('요일', fontsize=12)
plt.show()
