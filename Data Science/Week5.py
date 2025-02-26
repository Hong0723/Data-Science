import pandas as pd

df = pd.read_csv('GNI2014.csv')
df

df.shape[0]
df.shape[1]
df.loc[:, 'country']
df.loc[[1],['country','population', 'GNI']]
df.loc[df.continent == 'Europe', :]
df.loc[df.population >= 50000000, : ]
df2 = df.loc[df.continent == 'Europe', 'population'] 
df2.mean()

# 캐나다의 GNI 값 추출
canada_gni = df.loc[df['country'] == 'Canada', 'GNI'].values[0]

# 캐나다보다 GNI가 높은 국가 필터링
higher_gni_countries = df.loc[df['GNI'] > canada_gni, :]
higher_gni_countries

# 높은 GNI 국가들의 평균 GNI 계산
average_gni = higher_gni_countries['GNI'].mean()
average_gni




df.loc[df.continent == 'North America','GNI' ] * 1.1

Ec = df.loc[df.continent == 'Europe', :].copy()
Ec['GNI_per_capita'] = Ec.loc[:, 'GNI'] / Ec.loc[:, 'population']
Ec.loc[:, ['country', 'GNI_per_capita']]

PNP = Ec.GNI_per_capita
df['PNP'] = PNP
df
print(df.head(5))
