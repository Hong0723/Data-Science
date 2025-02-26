import folium
import geokakao as gk
import pandas as pd
import display as dm
import webbrowser

def showMap(map):
    map.save("map.html")
    webbrowser.open("map.html")

df_subway = pd.read_csv("subway_line_1_8_20231231.csv")
df_addr = pd.read_csv("seoul_subway_address_2023.csv")

time_zone =['t17_18시간대','t18_19시간대','t19_20시간대']
df_subway = df_subway.loc[df_subway.호선명 == 2]
df_subway = df_subway.loc[df_subway.승객유형 == '우대권']
df_subway = df_subway.groupby('역명')[time_zone].sum()
df_subway['탑승객수'] = df_subway.sum(axis=1)

df_merge = pd.merge(df_subway,df_addr, on ='역명', how ='inner')
gk.add_coordinates_to_dataframe(df_merge,'도로명주소')
df_merge.decimalLatitude = pd.to_numeric(df_merge.decimalLatitude)
df_merge.decimalLongitude = pd.to_numeric(df_merge.decimalLongitude)
print(df_merge[['역명','탑승객수','decimalLatitude', 'decimalLongitude']].head())
df_clean = df_merge.dropna(subset=['decimalLatitude', 'decimalLongitude'])
df_clean = df_clean.reset_index(drop =True)
center = df_clean[['decimalLatitude', 'decimalLongitude']].mean().to_list()

map = folium.Map(location=center,zomm_start=12)

for i in range(len(df_clean)) :
    folium.CircleMarker(location=[df_clean.loc[i,'decimalLatitude'],df_clean.loc[i,'decimalLongitude']],
                        radius=max(2,(df_clean.iloc[i]['탑승객수']/15000)), color ='red', stroke = False,fill=True,fill_opacity='50%').add_to(map)
    
    html_start = html = '<div\
        style ="\
            font-size: 12px;\
                color: blue;\
                    background-color:rgba(255,255,255,0.2);\
                        width:85px;\
                            text-align:left;\
                                margin:0px;\
                                    "><b>'
                                    html_end = '</b></div>'
                                    
                                    for i in range(len(df_clean)) :
                                        folium.Marker(locantion=[df_clean.loc[i,'decimalLatitude'], df_clean.loc[i,'decimalLongitude']],
                                                      icon=folium.DivIcon(
                                                          icon_anchor=(0,0),html =html_start + df_clean.loc[i,'역명']+html_end
                                                      )).add_to(map)
                                        showMap(map)