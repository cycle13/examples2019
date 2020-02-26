import folium

m = folium.Map(location=[29.488869,106.571034],
              zoom_start=7,
              control_scale=True)

ls = folium.PolyLine(locations=[[30.588869,105.671034],[29.488869,106.571034],[31.888869,104.971034],[30.588869,105.671034]],
                    color='blue')

ls.add_to(m)

m

m.save('map2.html')