from bokeh.io import curdoc, show
from bokeh.layouts import column
from bokeh.models import Div, CheckboxGroup, CheckboxButtonGroup, Button, OpenURL, Dropdown, Select
from functools import partial
import folium
import pandas as pd
import seaborn as sns
import numpy as np
from folium.plugins import MarkerCluster, HeatMap
from bokeh.plotting import gmap
import webbrowser

dfdic = {
    'FirstClass': ['FirstClass_Manipulated.csv', '#DAA520', 'credit-card-alt'],
    'SecondClass': ['SecondClass_Manipulated.csv', '#C0C0C0', 'money'],
    'ThirdClass': ['ThirdClass_Manipulated.csv', '#CD7F32', 'trash'],
    # 'CrossChannel':[ 'CrossChannel_Manipulated.csv', '#CD7F32', 'times'],
    'DeckCrew': ['DeckCrew_Manipulated.csv', "#1E90FF", 'glass'],
    'EngineeringCrew': ['EngineeringCrew_Manipulated.csv', '#663300', 'wrench'],
    'Officers': ['Officers_Manipulated.csv', '#000080', 'compass'],
    'OrchestraCrew': ['OrchestraCrew_Manipulated.csv', '#B22222', 'music'],
    'PostalCrew': ['PostalCrew_Manipulated.csv', '#708090', 'envelope'],
    'RestaurantCrew': ['RestaurantCrew_Manipulated.csv', '#808000', 'cutlery'],
    'VictuallingCrew': ['VictuallingCrew_Manipulated.csv', "#264348", 'ship']
}
#
passengerdictonary = {
    0: "FirstClass",
    1: 'SecondClass',
    2: 'ThirdClass'
}

crewdictonary = {
    3: 'DeckCrew',
    4: 'EngineeringCrew',
    5: 'Officers',
    6: 'OrchestraCrew',
    7: 'PostalCrew',
    8: 'RestaurantCrew',
    9: 'VictuallingCrew'
}

totaldictonary = {
    0: "FirstClass",
    1: 'SecondClass',
    2: 'ThirdClass',
    3: 'DeckCrew',
    4: 'EngineeringCrew',
    5: 'Officers',
    6: 'OrchestraCrew',
    7: 'PostalCrew',
    8: 'RestaurantCrew',
    9: 'VictuallingCrew'
}

dflist = [
    'FirstClass_Manipulated.csv',
    'SecondClass_Manipulated.csv',
    'ThirdClass_Manipulated.csv',
    # 'CrossChannel.csv',
    'DeckCrew_Manipulated.csv',
    'EngineeringCrew_Manipulated.csv',
    'Officers_Manipulated.csv',
    'OrchestraCrew_Manipulated.csv',
    'PostalCrew_Manipulated.csv',
    'RestaurantCrew_Manipulated.csv',
    'VictuallingCrew_Manipulated.csv'
]

colordic = {
    True: 'green',
    False: "red"}

maptype = "MarkerMap"

dlist = list(dfdic)

passengerlist = [dlist[0],
                 dlist[1],
                 dlist[2],
                 # dlist[3]
                 ]

crewlist = [dlist[4],
            dlist[5],
            dlist[6],
            dlist[7],
            dlist[6],
            dlist[7],
            dlist[8]]

totallist = [dlist[0],
             dlist[1],
             dlist[2],
             dlist[4],
             dlist[5],
             dlist[6],
             dlist[7],
             dlist[6],
             dlist[7],
             dlist[8]]
LABELS = ["Passengers", "Crew"]

d1 = CheckboxButtonGroup(labels=passengerlist)
d2 = CheckboxButtonGroup(labels=crewlist)
d3 = CheckboxButtonGroup(labels=totallist)
c = CheckboxButtonGroup(labels=LABELS, active=[0, 1])
d1.visible = False
d2.visible = False


def checkbox_changed(attr, old, new):
    if new == [0]:
        d1.visible = True
        d2.visible = False
        d3.visible = False
    elif new == [1]:
        d2.visible = True
        d1.visible = False
        d3.visible = False
    elif new == [0, 1]:
        d1.visible = False
        d2.visible = False
        d3.visible = True
    elif new == []:
        d1.visible = False
        d2.visible = False
        d3.visible = False


c.on_change('active', checkbox_changed)


def drawmap(lijst):
    map1 = folium.Map(
        location=[50, 0],
        tiles='stamentoner',
        zoom_start=1,
    )

    print(type(lijst))
    print(lijst)
    teller = 0



    for naam in lijst:
        print(naam)
        print(maptype)
        path = "Manipulated/" + naam + "_Manipulated.csv"
        df = pd.read_csv(path, index_col=0)
        df = df.dropna(subset=["latitude"]).reset_index(drop=True)
        dflist[teller] = df

        unique = df.groupby('Hometown')['Name'].nunique()
        # print (unique)
        title = dlist[teller]
        if maptype == "MarkerMap":
            feature_group = folium.FeatureGroup(title)
            marker_cluster = MarkerCluster().add_to(map1)

            for row in df.itertuples():
                folium.Marker(
                    location=[row.latitude, row.longitude],
                    radius=1,
                    popup=(str(row.Name + " :"+row.Position)),
                    # fill=True, # Set fill to True
                    # fill_color=color_producer(el),
                    color=dfdic[naam][1],
                    icon=folium.Icon(color=colordic[row.Survived],
                                     icon_color='white',
                                     icon=dfdic[naam][2],
                                     angle=0,
                                     prefix='fa')).add_to(feature_group)

            feature_group.add_to(map1)
            teller = teller + 1

        if maptype == "ClusterMap":
            feature_group = folium.FeatureGroup(title)
            marker_cluster = MarkerCluster().add_to(map1)

            for row in df.itertuples():
                folium.Marker(
                    location=[row.latitude, row.longitude],
                    radius=1,
                    popup=(str(row.Name + " :"+row.Position)),
                    # fill=True, # Set fill to True
                    # fill_color=color_producer(el),
                    color=dfdic[naam][1],
                    icon=folium.Icon(color=colordic[row.Survived],
                                     icon_color='white',
                                     icon=dfdic[naam][2],
                                     angle=0,
                                     prefix='fa')).add_to(marker_cluster)

            feature_group.add_to(map1)
            teller = teller + 1

        if maptype == "HeatMap":
            heat_data = [[row['latitude'], row['longitude']] for index, row in df.iterrows()]
            # Plot it on the map
            HeatMap(heat_data).add_to(map1)

    folium.LayerControl().add_to(map1)
    map1.save(r'C:\Users\ruben\Data\Titinic_Locations.html')
    print("saving done...")


def forwardData(list, data):
    lijst = []
    if list == 'passengerlist':
        for a in data:
            print(a)
            lijst.append(passengerdictonary[a])

    if list == 'crewlist':
        for a in data:
            print(a)
            lijst.append(crewdictonary[a])

    if list == 'totallist':
        for a in data:
            print(a)
            lijst.append(totaldictonary[a])

    drawmap(lijst)

    # print(Output)
    # Output = [b for b in data if
    #           all(a not in b for a in dflijst)]
    # # selectie = [x for x in dflijst if x[0] in data]
    # selectie = dflijst[data]
    # print(Output)


def getPassengers(attr, old, new):
    print(type(new))
    # new = [new]
    forwardData("passengerlist", new)


def getCrew(attr, old, new):
    print(type(new))
    # new = [new]
    forwardData("crewlist", new)


def getBoth(attr, old, new):
    print(type(new))
    # new = [new]
    forwardData("totallist", new)


def showMap():
    url = r"C:\Users\ruben\Data\Titinic_Locations.html"
    webbrowser.open_new_tab(url)

d1.on_change('active', getPassengers)
d2.on_change('active', getCrew)
d3.on_change('active', getBoth)

# menu = [("MarkerMap", "item_1"), ("ClusterMap", "item_2"), ("HeatMap", "item_3")]
#
# dropdown = Dropdown(label="Choose map type", menu=menu)
#
# dropdown.on_change('value', getType)
dropdown = Select(title='Choose map type', options=['MarkerMap', 'ClusterMap', 'HeatMap'], value='MarkerMap')


def handler(attr, old, new):
    global maptype
    maptype = new

dropdown.on_change('value', handler)

button = Button(label="ShowMap", button_type="success")
button.on_click(showMap)
# mapa = Div(text="<iframe src="r'C:\Users\ruben\Data\Titinic_Locations.html'" style='min-width:calc(100vw - 26px); height: 500px'><iframe>")

# show(mapa)
curdoc().add_root(column(c, dropdown,  d1, d2, d3, button))
