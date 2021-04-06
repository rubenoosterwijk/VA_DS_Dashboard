# importeren libraries
from bokeh.io import curdoc
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import widgetbox, row, column
from bokeh.models.widgets import Panel, Tabs, Paragraph, Div, DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource, Select, Dropdown
from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import dim
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


hv.extension('bokeh')
from bokeh.io import show
from bokeh.models import CheckboxGroup, CustomJS

import ipywidgets
from bokeh.io import push_notebook
from bokeh.models import Range1d



# Header en inleiding
titel1 = Div(text="<h1"">Hoofdpagina correlatie ticketprijzen titanic en gdp per regio""</h1>", width=800, height=50)
text1 = Div(
    text="<h4"">Op dit dashboard geven we weer wat de correlatie is tussen de consumenten van de ticketprijzen, en het gdp per regio""</h4>",
    width=800, height=50)
text21 = Div(text="<h3"">Inhoudsopgave:""</h3>",width=800, height=50)
text22 = Div(text="<h4"">Hoofdstuk 1: Dataverzameling""</h4>",width=800, height=30)
text23 = Div(text="<h4"">Hoofdstuk 2: 1-Dimensionele visualisaties""</h4>",width=800, height=30)
text24 = Div(text="<h4"">Hoofdstuk 3: 2-Dimensionele visualisaties""</h4>",width=800, height=30)
text25 = Div(text="<h4"">Hoofdstuk 4: Kaart""</h4>",width=800, height=30)
text26 = Div(text="<h4"">Hoofdstuk 5: Lineaire regressie""</h4>",width=800, height=30)
text27 = Div(text="<h4"">Hoofdstuk 6: Bronvermelding""</h4>",width=800, height=30)

# Hoofdstuk 1 Dataverzameling
titel2 = Div(text="<h2"">Hoofdstuk 1: Data-analyse en verzameling""</h2>", width=800, height=50)
text2 = Div(
    text="<h4"">Om dit weer te geven hebben we eerst het GDP per regio van het dichtsbijstaande jaar (1911) moeten vinden""</h4>",
    width=800, height=50)

d = {'Region': ['London', 'Rest South East', 'East Anglia', 'South West', 'West Midlands',
                'East Midlands', 'North West', 'Yorks & Humb', 'North', 'Wales', 'Scotland', 'Ireland'],
     'GDP': [416.0, 313.8, 48.8, 120.9, 158.0, 130.2, 323.1, 185.9, 130.2, 116.5, 240.0, 146.8]}
df = pd.DataFrame(data=d)
Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns]  # bokeh columns
data_table = DataTable(columns=Columns, source=ColumnDataSource(df))  # bokeh table

text3 = Div(
    text="<h4"">Daarnaast hebben we de bemanning en passagiers opgedeeld in werkfunctie en klasse. Hieronder volgt een voorbeeld van de passagiers uit de eerste klasse""</h4>",
    width=800, height=50)
FirstClass = pd.read_csv('Passengers\FirstClass.csv')
Columns = [TableColumn(field=Ci, title=Ci) for Ci in FirstClass.columns]  # bokeh columns
data_table2 = DataTable(columns=Columns, source=ColumnDataSource(FirstClass))  # bokeh table

# Hoofdstuk 2 Histogram en 1-D visualisatie
titel3 = Div(text="<h2"">Hoofdstuk 2: Barplots en boxplots""</h2>", width=800, height=50)
# h = figure(x_axis_label="Region", y_axis_label="GDP")
# h.vbar(x='Region', y='GDP', source=source)


# inladen van de dataframes
UKRegio = pd.read_csv(r"Data K\UKRegio.csv")
UKRegio1eKlas = pd.read_csv(r"Data K\UKRegio1eKlas.csv")
UKRegio2eKlas = pd.read_csv(r"Data K\UKRegio2eKlas.csv")
UKRegio3eKlas = pd.read_csv(r"Data K\UKRegio3eKlas.csv")

Klas1 = pd.read_csv(r"Manipulated\FirstClass_Manipulated.csv")
Klas2 = pd.read_csv(r"Manipulated\SecondClass_Manipulated.csv")
Klas3 = pd.read_csv(r"Manipulated\ThirdClass_Manipulated.csv")

# Barchart
x_barRegio = UKRegio['Regio'].unique()
y_barRegio = UKRegio['Regio'].value_counts()

x_barRegio1 = UKRegio1eKlas['Regio'].unique()
y_barRegio1 = UKRegio1eKlas['Regio'].value_counts()

x_barRegio2 = UKRegio2eKlas['Regio'].unique()
y_barRegio2 = UKRegio2eKlas['Regio'].value_counts()

x_barRegio3 = UKRegio3eKlas['Regio'].unique()
y_barRegio3 = UKRegio3eKlas['Regio'].value_counts()

x1 = x_barRegio
y1 = y_barRegio

# Create ColumnDataSource: sourceRegio
sourceRegio = ColumnDataSource(data={
    'x': x_barRegio,
    'y': y_barRegio
})

sourceRegio1 = ColumnDataSource(data={
    'x': x_barRegio1,
    'y': y_barRegio1
})

sourceRegio2 = ColumnDataSource(data={
    'x': x_barRegio2,
    'y': y_barRegio2
})

sourceRegio3 = ColumnDataSource(data={
    'x': x_barRegio3,
    'y': y_barRegio3
})

# Maak een nieuwe plot
bar_chart = figure(x_range=x_barRegio, title='BarPlot Alle Klasse', x_axis_label='Regio in engeland en GDP in Pond',
                   y_axis_label='Aantal tickets', plot_height=300, plot_width=1500)

# Voeg de barchart toe
bar_chart.vbar('x', top='y', source=sourceRegio, color='blue', width=0.5)
bar_chart.y_range.start = 0

bar_chart1 = figure(x_range=x_barRegio, title='BarPlot 1e Klasse', x_axis_label='Regio in engeland en GDP in Pond',
                    y_axis_label='Aantal tickets', plot_height=300, plot_width=1500)

# Voeg de barchart toe
bar_chart1.vbar('x', top='y', source=sourceRegio1, color='blue', width=0.5)
bar_chart1.y_range.start = 0

bar_chart420 = figure(x_range=x_barRegio, title='BarPlot 2e Klasse', x_axis_label='Regio in engeland en GDP in Pond',
                      y_axis_label='Aantal tickets', plot_height=300, plot_width=1500)

# Voeg de barchart toe
bar_chart420.vbar('x', top='y', source=sourceRegio2, color='blue', width=0.5)
bar_chart420.y_range.start = 0

bar_chart3 = figure(x_range=x_barRegio, title='BarPlot 3e Klasse', x_axis_label='Regio in engeland en GDP in Pond',
                    y_axis_label='Aantal tickets', plot_height=300, plot_width=1500)

# Voeg de barchart toe
bar_chart3.vbar('x', top='y', source=sourceRegio3, color='blue', width=0.5)
bar_chart3.y_range.start = 0

bar_chart.visible = False
bar_chart1.visible = False
bar_chart420.visible = False
bar_chart3.visible = False

Klas1["Age"] = pd.to_numeric(Klas1["Age"], errors='coerce')
Klas2["Age"] = pd.to_numeric(Klas2["Age"], errors='coerce')
Klas3["Age"] = pd.to_numeric(Klas3["Age"], errors='coerce')


# boxplot1 = sns.catplot(x="country", y="Age", hue="Survived", data=Klas1, palette="Set3", kind="box")
# boxplot2 = sns.catplot(x="country", y="Age", hue="Survived", data=Klas2, palette="Set3", kind="box")
# boxplot3 = sns.catplot(x="country", y="Age", hue="Survived", data=Klas3, palette="Set3", kind="box")

# boxplot1 = sns.catplot(x="Age", y="Survived", kind="box", orient="h", height=1.5, aspect=4,
#                 data=Klas1)
# boxplot2 = sns.catplot(x="Age", y="Survived", kind="box", orient="h", height=1.5, aspect=4,
#                 data=Klas2)
# boxplot3 = sns.catplot(x="Age", y="Survived", kind="box", orient="h", height=1.5, aspect=4,
#                 data=Klas3)


# Define a callback function: update_plot
def update_bar_chart(event):
    new = event.item

    # If all laat alle klasse zien
    if new == 'All':
        bar_chart.visible = True
        bar_chart1.visible = False
        bar_chart420.visible = False
        bar_chart3.visible = False
    # Elif naar 1e klas
    elif new == '1e Klass':
        bar_chart.visible = False
        bar_chart1.visible = True
        bar_chart420.visible = False
        bar_chart3.visible = False
    elif new == '2e Klass':
        bar_chart.visible = False
        bar_chart1.visible = False
        bar_chart420.visible = True
        bar_chart3.visible = False
    elif new == '3e Klass':
        bar_chart.visible = False
        bar_chart1.visible = False
        bar_chart420.visible = False
        bar_chart3.visible = True


# Create a dropdown Select widget: select
selectRegio = Dropdown(label="Maak keuze uit de klasse", menu=["All", "1e Klass", "2e Klass", "3e Klass"])

# Attach the update_plot callback to the 'value' property of select
selectRegio.on_click(update_bar_chart)

# Hoofdstuk 3 Scatter en 2-D visualisatie
titel4 = Div(text="<h2"">Hoofdstuk 3: Scatterplots en 2-D visualisaties""</h2>", width=800, height=50)
text4 = Div(text="<h4"">Hieronder volgt scatterplot van prijs per ticket tegenover de GDP per capita""</h4>", width=800,
            height=50)
# dataframes
FareVsGDP = pd.read_csv('Data K\FareVsGDP.csv')
FareVsGDP1 = pd.read_csv('Data K\FareVsGDP1eKlass.csv')
FareVsGDP2 = pd.read_csv('Data K\FareVsGDP2eKlass.csv')
FareVsGDP3 = pd.read_csv('Data K\FareVsGDP3eKlass.csv')

# dropdown
menu = [("Class 1", "Class_1"), ("Class 2", "Class_2"), None, ("Class 3", "Class_3")]

# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x': FareVsGDP['GDP_Per_Capita'],
    'y': FareVsGDP['Fare']
})

# Create a new plot: plot
plot = figure()

# Add circles to the plot
plot.circle('x', 'y', source=source)


# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If all laat alle klasse zien
    if new == 'All':
        source.data = {
            'x': FareVsGDP['GDP_Per_Capita'],
            'y': FareVsGDP['Fare']
        }
    # Elif naar 1e klas
    elif new == '1e Klass':
        source.data = {
            'x': FareVsGDP1['GDP_Per_Capita'],
            'y': FareVsGDP1['Fare']
        }
    elif new == '2e Klass':
        source.data = {
            'x': FareVsGDP2['GDP_Per_Capita'],
            'y': FareVsGDP2['Fare']
        }
    elif new == '3e Klass':
        source.data = {
            'x': FareVsGDP3['GDP_Per_Capita'],
            'y': FareVsGDP3['Fare']
        }


# Create a dropdown Select widget: select
select = Select(title="Keuzemenu", options=["All", "1e Klass", "2e Klass", "3e Klass"], value="All")

# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)

# Hoofdstuk 4
titel5 = Div(text="<h2"">Hoofdstuk 4: Foliumkaart""</h2>", width=800, height=50)
text5 = Div(text="<h4"">Hieronder volgt onze kaart""</h4>", width=800, height=50)

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
    0: 'DeckCrew',
    1: 'EngineeringCrew',
    2: 'Officers',
    3: 'OrchestraCrew',
    4: 'PostalCrew',
    5: 'RestaurantCrew',
    6: 'VictuallingCrew'
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

crewlist = [dlist[3],
            dlist[4],
            dlist[5],
            dlist[6],
            dlist[7],
            dlist[8],
            dlist[9]]

totallist = [dlist[0],
             dlist[1],
             dlist[2],
             dlist[3],
             dlist[4],
             dlist[5],
             dlist[6],
             dlist[7],
             dlist[8],
             dlist[9]]
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
        title = naam
        print (title, "XD")
        if maptype == "MarkerMap":
            feature_group = folium.FeatureGroup(title)
            marker_cluster = MarkerCluster().add_to(map1)

            for row in df.itertuples():
                folium.Marker(
                    location=[row.latitude, row.longitude],
                    radius=1,
                    popup=(str(row.Name + " :" + row.Position)),
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
                    popup=(str(row.Name + " :" + row.Position)),
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


# Hoofdstuk 5
titel6 = Div(text="<h2"">Hoofdstuk 5: Regressiemodel""</h2>", width=800, height=50)
text6 = Div(text="<h4"">Hieronder eerst de data die in het regressiemodel geladen kan worden.""</h4>", width=800,
            height=50)
text7 = Div(
    text="<h4"">Engeland is hier weggelaten aangezien hier de boot is vertrokken en er dus duidelijk een meerderheid aan passagiers is.""</h4>",
    width=800, height=50)

# Toe te voegen dataframes
FareVsGDPCountry = pd.read_csv('Data K\FareVsGDPCountry.csv')

x_barLand = FareVsGDPCountry['Land'].unique()
y_barLand = FareVsGDPCountry['Land'].value_counts()

# Maak een nieuwe plot
bar_chart2 = figure(x_range=x_barLand, title='BarPlot Landen', x_axis_label='Land', y_axis_label='Aantal tickets',
                    plot_height=300, plot_width=1500)

# Voeg de barchart toe
bar_chart2.vbar(x_barLand, top=y_barLand, color='blue', width=0.5)
bar_chart2.y_range.start = 0

text8 = Div(text="<h4"">Aangezien het schip uit de UK vertrekt is het logisch dat hier meer mensen vandaan zullen komen, daarom volgt hieronder de dataset zonder de UK.""</h4>", width=800, height=50)

FareVsGDPNoUK = FareVsGDPCountry[FareVsGDPCountry.Land != 'UK']

x_barLandNUK = FareVsGDPNoUK['Land'].unique()
y_barLandNUK = FareVsGDPNoUK['Land'].value_counts()

# Maak een nieuwe plot
bar_chart3 = figure(x_range=x_barLandNUK, title='BarPlot Landen', x_axis_label='Land', y_axis_label='Aantal tickets',
                    plot_height=300, plot_width=1500)

# Voeg de barchart toe
bar_chart3.vbar(x_barLandNUK, top=y_barLandNUK, color='blue', width=0.5)
bar_chart3.y_range.start = 0

text9 = Div(text="<h4"">Nu volgt ons regressiemodel.""</h4>", width=800, height=50)

FareVsGDPNoUK = FareVsGDPNoUK.sample(frac=1).reset_index(drop=True)

model_x = FareVsGDPNoUK['GDP_Per_Capita']
model_y = FareVsGDPNoUK['Fare']

model_x = np.array(model_x).reshape((-1, 1))
model_y = np.array(model_y).reshape((-1, 1))

# Split the data into training/testing sets
diabetes_X_train = model_x[:-50]
diabetes_X_test = model_x[-50:]

# Split the targets into training/testing sets
diabetes_y_train = model_y[:-50]
diabetes_y_test = model_y[-50:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

diabetes_X_test = np.array(diabetes_X_test).reshape((-1, 1))
diabetes_y_test = np.array(diabetes_y_test).reshape((-1, 1))
my_array1 = np.array(diabetes_X_test)
my_array2 = np.array(diabetes_y_test)
diabetes_X_test12 = pd.DataFrame(my_array1, columns = ['X'])
diabetes_y_test12 = pd.DataFrame(my_array2, columns = ['Y'])

diabetes_y_pred = np.array(diabetes_y_pred).reshape((-1, 1))
my_array3 = np.array(diabetes_y_pred)
diabetes_y_pred12 = pd.DataFrame(my_array3, columns = ['Y'])





FareVsGDPCountry2 = FareVsGDPCountry.sample(frac=1).reset_index(drop=True)

model_x1 = FareVsGDPCountry2['GDP_Per_Capita']
model_y1 = FareVsGDPCountry2['Fare']

model_x1 = np.array(model_x1).reshape((-1, 1))
model_y1 = np.array(model_y1).reshape((-1, 1))

# Split the data into training/testing sets
diabetes_X_train1 = model_x1[:-50]
diabetes_X_test1 = model_x1[-50:]

# Split the targets into training/testing sets
diabetes_y_train1 = model_y1[:-50]
diabetes_y_test1 = model_y1[-50:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train1, diabetes_y_train1)

# Make predictions using the testing set
diabetes_y_pred1 = regr.predict(diabetes_X_test1)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test1, diabetes_y_pred1))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test1, diabetes_y_pred1))

diabetes_X_test1 = np.array(diabetes_X_test1).reshape((-1, 1))
diabetes_y_test1 = np.array(diabetes_y_test1).reshape((-1, 1))
my_array11 = np.array(diabetes_X_test1)
my_array21 = np.array(diabetes_y_test1)
diabetes_X_test11 = pd.DataFrame(my_array11, columns = ['X'])
diabetes_y_test11 = pd.DataFrame(my_array21, columns = ['Y'])

diabetes_y_pred11 = np.array(diabetes_y_pred1).reshape((-1, 1))
my_array31 = np.array(diabetes_y_pred1)
diabetes_y_pred11 = pd.DataFrame(my_array31, columns = ['Y'])



plotRegression2 = figure()
plotRegression2.circle(diabetes_X_test11['X'], diabetes_y_test11['Y'])
plotRegression2.line(diabetes_X_test11['X'], diabetes_y_pred11['Y'], line_width=2)



plotRegression = figure()
plotRegression.circle(diabetes_X_test12['X'], diabetes_y_test12['Y'])
plotRegression.line(diabetes_X_test12['X'], diabetes_y_pred12['Y'], line_width=2)




plotRegression.visible = False
plotRegression2.visible = False



# Define a callback function: update_plot
def update_bar_chart(event):
    new = event.item

    # If all laat alle klasse zien
    if new == 'Zonder UK':
        plotRegression.visible = True
        plotRegression2.visible = False
    # Elif naar 1e klas
    elif new == 'Met UK':
        plotRegression.visible = False
        plotRegression2.visible = True

# Create a dropdown Select widget: select
selectUK = Dropdown(label="Maak keuze uit de landen", menu=["Zonder UK", "Met UK"])

# Attach the update_plot callback to the 'value' property of select
selectUK.on_click(update_bar_chart)

#Hoofdstuk 6 bronvermelding
titel7 = Div(text="<h2"">Hoofdstuk 6: Bronvermelding""</h2>", width=800, height=50)
text11 = Div(text="<h4"">GDP van engelse regio's anno 1911: http://eprints.lse.ac.uk/22557/1/0304Crafts.pdf.""</h4>", width=800,height=50)
text12 = Div(text="<h4"">GDP per capita van alle landen anno 1910: https://www.oecd-ilibrary.org/economics/how-was-life_9789264214262-en.""</h4>", width=800,height=50)
text13 = Div(text="<h4"">Alle bemanningsleden van de titanic: https://en.wikipedia.org/wiki/Crew_of_the_Titanic.""</h4>", width=800,height=50)
text14 = Div(text="<h4"">Alle passagiers en afkomsten van de titanic https://en.wikipedia.org/wiki/Passengers_of_the_Titanic.""</h4>", width=800,height=50)


output_file("Hoofdpagina.html", title="Hoofdpagina Dashboard V.A.")
# boxplot1, boxplot2, boxplot3
# Creeer de kolommen voor de layout
Home = column(titel1, text1, text21, text22, text23, text24, text25, text26,text27)
h1 = column(titel2, text2, data_table, text3, data_table2)
h2 = column(titel3, selectRegio, bar_chart, bar_chart3, bar_chart420, bar_chart1)
h3 = column(titel4, text4, select, plot)
h4 = column(titel5, text5, c, dropdown, d1, d2, d3, button)
h5 = column(titel6, text6, bar_chart2, text8, bar_chart3, text9, selectUK, plotRegression, plotRegression2)
h6 = column(titel7, text11, text12, text13, text14)

# Maak de tabs
# Create tab1 from plot p1: tab1
tab1 = Panel(child=Home, title='Hoofdpagina')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=h1, title='Hoofdstuk 1')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=h2, title='Hoofdstuk 2')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=h3, title='Hoofdstuk 3')

# Create tab4 from plot p4: tab4
tab5 = Panel(child=h4, title='Hoofdstuk 4')

# Create tab4 from plot p4: tab4
tab6 = Panel(child=h5, title='Hoofdstuk 5')

# Create tab4 from plot p4: tab4
tab7 = Panel(child=h6, title='Bronvermelding')

# Create layout and add to current document

# layout = column(titel1, text1, titel2, text2, data_table, text3, data_table2, titel3, titel4, text4, select, plot)
layout = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6, tab7])
curdoc().add_root(layout)
