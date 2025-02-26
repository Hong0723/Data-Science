import folium
import geokakao as gk
import webbrowser

def showMap(map):
    map.save("map.html")
    webbrowser.open("map.html")


#### 코드 9-1 ####

# 지도 중심의 좌표를 알 때
loc = [37.54, 127.05] # 위도와 경도
map = folium.Map(location=loc) # 지도 객체 생성
showMap(map) # 웹 브라우저에 지도 표시

# 지도 중심의 주소를 알 때
loc = gk.convert_address_to_coordinates('경기 용인시 수지구 죽전로 152')
loc                                             # 지도의 중심좌표
map = folium.Map(location=loc, zoom_start=16) # 지도 객체 생성
showMap(map) # 웹브라우저에 지도 표시


#### 코드 9-2 #### 지도 위에 마커, 텍스트 표시

# (1) 기본 마커의 표시
loc = gk.convert_address_to_coordinates('서울 종로구 사직로 161')
loc
map = folium.Map(location=loc, zoom_start=16) # 지도 객체 생성
folium.Marker(location=loc,     # 기본 마커 표시
              popup = '경복궁').add_to(map)
showMap(map)

#### 코드 9-3 ####
 
# (2) 마커 아이콘의 색과 모양을 변경하는 예
loc = gk.convert_address_to_coordinates('강원특별자치도 강릉시 창해로 514')
loc
map = folium.Map(location=loc, zoom_start=13) # 지도 객체 생성
folium.Marker(location=loc,icon=folium.Icon(color='red', icon='star')).add_to(map)
showMap(map)

#### 코드 9-4 ####

# (3) 마커 위치에 텍스트 추가하기
html_start = html = '<div\
                    style="\
                        font-size: 12px;\
                        color: blue;\
                            background-color:rgbaa(255,255,255,0.2);\
                                width:85px;\
                                    text-align:left;\
                                        margin:0px;\
                                            "><b>'
                                            html_end = '</b></div>'
folium.Marker(location=loc,
              icon=folium.DivIcon(
                  icon_anchor=(0,0),
                  html = html_start +'경포해수욕장' + html_end
              )).add_to(map)
showMap(map)

#### 코드 9-5 ####
# (4) 다수의 아이콘 표시

import folium
import geokakao as gk
import pandas as pd

# 관광지 정보를 데이터프레임으로 저장
names = ['용두암','성산일출봉','정방폭포',
         '중문관광단지','한라산1100고지','차귀도']
addr = ['제주시 용두암길 15',
        '서귀포시 선산읍 성산리',
        '서귀포시 동홍동 299-3',
        '서귀포시 중문동 2624-1',
        '서귀포시 색달동 산1-2',
        '제주시 한경면 고산리 산 117']
dict = {"names" : names, "addr" : addr}
df= pd.DataFrame(dict) 
df

# 관광지의 좌표를 df에 추가
gk.add_coordinates_to_dataframe(df,'addr')
df.dtypes

# 문자열 좌표값을 숫자로 변환
df.decimalLatitude = pd.to_numeric(df.decimalLatitude)
df.decimalLongitude = pd.to_numeric(df.decimalLongitude)
df.dtypes

# 지도의 중심점 계산
center = [df.decimalLatitude.mean(), df.decimalLongitude.mean()]
center

# 지도 객체 생성
map = folium.Map(location=center, zoom_start = 10)

# 지도에 마커 추가 
for i in range(len(df)):
    folium.Marker(location =[df.iloc[i,2], df.iloc[i,3]],
                  icon=folium.Icon(color='red', icon= 'star')).add_to(map)
    showMap(map)


#### 지도 위에 데이터 표시 ####

# 코드 9-6 #
import folium
import pandas as pd
df = pd.read_csv('wind.csv')
df

df = df.sample(50,random_state=123)

# 지도의 중심점 구하기
center = [df.lat.mean(), df.lon.mean()]

# 지도 가져오기
map = folium.Map(location= center, zoom_start=5)
showMap(map)

# 측정위치에 마커 표시하기
for i in range(len(df)):
    folium.Marker(location =[df.lat.iloc[i], df.lon.iloc[i]],
                  icon = folium.Icon(color='blue',icon ='flag')).add_to(map)
    showMap(map)
    
# 풍속을 원의 크기로 표시하기
map = folium.Map(location= center, zoom_start=5) # 마커 없는 지도
for i in range(len(df)) :
    folium.CircleMarker(location=[df.lat.iloc[i],df.lon.iloc[i]],
                        radius= (df.spd.iloc[i]**0.5)*2, # 원의 반지름
                        color='red', # 원의 색
                        stroke=False, # 윤곽선 없음
                        fill = True, # 원의 내부 색
                        fill_opacity='50%').add_to(map) # 원의 내부색 투명도
    showMap(map)