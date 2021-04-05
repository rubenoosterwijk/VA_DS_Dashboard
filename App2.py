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

from bokeh.io import show
from bokeh.models import CheckboxGroup, CustomJS

import ipywidgets
from bokeh.io import push_notebook
from bokeh.models import Range1d



#Header en inleiding
titel1 = Div(text = "<h1"">Hoofdpagina correlatie ticketprijzen titanic en gdp per regio""</h1>", width = 800, height = 50)
text1 = Div(text = "<h4"">Op dit dashboard geven we weer wat de correlatie is tussen de consumenten van de ticketprijzen, en het gdp per regio""</h4>", width = 800, height = 50)



#Hoofdstuk 1 Dataverzameling
titel2 = Div(text = "<h2"">Hoofdstuk 1: Data-analyse en verzameling""</h2>", width = 800, height = 50)
text2 = Div(text = "<h4"">Om dit weer te geven hebben we eerst het GDP per regio van het dichtsbijstaande jaar (1911) moeten vinden""</h4>", width = 800, height = 50)

d = {'Region': ['London', 'Rest South East', 'East Anglia', 'South West', 'West Midlands',
                'East Midlands', 'North West', 'Yorks & Humb', 'North', 'Wales', 'Scotland', 'Ireland'],
     'GDP': [416.0, 313.8, 48.8, 120.9, 158.0, 130.2, 323.1, 185.9, 130.2, 116.5, 240.0, 146.8]}
df = pd.DataFrame(data=d)
Columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] # bokeh columns
data_table = DataTable(columns=Columns, source=ColumnDataSource(df)) # bokeh table

text3 = Div(text = "<h4"">Daarnaast hebben we de bemanning en passagiers opgedeeld in werkfunctie en klasse. Hieronder volgt een voorbeeld van de passagiers uit de eerste klasse""</h4>", width = 800, height = 50)
FirstClass = pd.read_csv('Passengers\FirstClass.csv')
Columns = [TableColumn(field=Ci, title=Ci) for Ci in FirstClass.columns] # bokeh columns
data_table2 = DataTable(columns=Columns, source=ColumnDataSource(FirstClass)) # bokeh table



# Hoofdstuk 2 Histogram en 1-D visualisatie
titel3 = Div(text = "<h2"">Hoofdstuk 2: Barplots en boxplots""</h2>", width = 800, height = 50)
#h = figure(x_axis_label="Region", y_axis_label="GDP")
#h.vbar(x='Region', y='GDP', source=source)


#inladen van de dataframes
UKRegio = pd.read_csv(r"Data K\UKRegio.csv")
UKRegio1eKlas = pd.read_csv(r"Data K\UKRegio1eKlas.csv")
UKRegio2eKlas = pd.read_csv(r"Data K\UKRegio2eKlas.csv")
UKRegio3eKlas = pd.read_csv(r"Data K\UKRegio3eKlas.csv")


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
    'x' : x_barRegio,
    'y' : y_barRegio
})

# Maak een nieuwe plot
bar_chart = figure(x_range='x', source=sourceRegio, title='BarPlot',x_axis_label='Regio in engeland en GDP in Pond', y_axis_label='Aantal tickets', plot_height=300, plot_width = 1500)


# Voeg de barchart toe
bar_chart.vbar('x', top='y', source=sourceRegio, color='blue', width=0.5)
bar_chart.y_range.start = 0

# Define a callback function: update_plot
def update_bar_chart(attr, old, new):
    # If all laat alle klasse zien
    if new == 'All':
        source.data = {
            'x' : x_barRegio,
            'y' : y_barRegio
        }
    # Elif naar 1e klas
    elif new == '1e Klass':
        source.data = {
            'x' : x_barRegio1,
            'y' : y_barRegio1
        }
    elif new == '2e Klass':
        source.data = {
            'x' : x_barRegio2,
            'y' : y_barRegio2
        }
    elif new == '3e Klass':
        source.data = {
            'x' : x_barRegio3,
            'y' : y_barRegio3
        }

# Create a dropdown Select widget: select
selectRegio = Select(title="Maak keuze uit de klasse", options=["All", "1e Klass", "2e Klass", "3e Klass"], value="All")


# Attach the update_plot callback to the 'value' property of select
selectRegio.on_change('value', update_bar_chart)








# Hoofdstuk 3 Scatter en 2-D visualisatie
titel4 = Div(text = "<h2"">Hoofdstuk 3: Scatterplots en 2-D visualisaties""</h2>", width = 800, height = 50)
text4 = Div(text = "<h4"">Hieronder volgt scatterplot van prijs per ticket tegenover de GDP per capita""</h4>", width = 800, height = 50)
#dataframes
FareVsGDP = pd.read_csv('Data K\FareVsGDP.csv')
FareVsGDP1 = pd.read_csv('Data K\FareVsGDP1eKlass.csv')
FareVsGDP2 = pd.read_csv('Data K\FareVsGDP2eKlass.csv')
FareVsGDP3 = pd.read_csv('Data K\FareVsGDP3eKlass.csv')

#dropdown
menu = [("Class 1", "Class_1"), ("Class 2", "Class_2"), None, ("Class 3", "Class_3")]



# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : FareVsGDP['GDP_Per_Capita'],
    'y' : FareVsGDP['Fare']
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
            'x' : FareVsGDP['GDP_Per_Capita'],
            'y' : FareVsGDP['Fare']
        }
    # Elif naar 1e klas
    elif new == '1e Klass':
        source.data = {
            'x' : FareVsGDP1['GDP_Per_Capita'],
            'y' : FareVsGDP1['Fare']
        }
    elif new == '2e Klass':
        source.data = {
            'x' : FareVsGDP2['GDP_Per_Capita'],
            'y' : FareVsGDP2['Fare']
        }
    elif new == '3e Klass':
        source.data = {
            'x' : FareVsGDP3['GDP_Per_Capita'],
            'y' : FareVsGDP3['Fare']
        }

# Create a dropdown Select widget: select
select = Select(title="Keuzemenu", options=["All", "1e Klass", "2e Klass", "3e Klass"], value="All")


# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)












# Extra voorbeeldcode
cats = list("abcdef")
yy = np.random.randn(2000)
g = np.random.choice(cats, 2000)
for i, l in enumerate(cats):
    yy[g == l] += i // 2
df = pd.DataFrame(dict(score=yy, group=g))

# find the quartiles and IQR for each category
groups = df.groupby('group')
q1 = groups.quantile(q=0.25)
q2 = groups.quantile(q=0.5)
q3 = groups.quantile(q=0.75)
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

# find the outliers for each category
def outliers(group):
    cat = group.name
    return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
out = groups.apply(outliers).dropna()

# prepare outlier data for plotting, we need coordinates for every outlier.
if not out.empty:
    outx = list(out.index.get_level_values(0))
    outy = list(out.values)

p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

# if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
qmin = groups.quantile(q=0.00)
qmax = groups.quantile(q=1.00)
upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]

# stems
p.segment(cats, upper.score, cats, q3.score, line_color="black")
p.segment(cats, lower.score, cats, q1.score, line_color="black")

# boxes
p.vbar(cats, 0.7, q2.score, q3.score, fill_color="#E08E79", line_color="black")
p.vbar(cats, 0.7, q1.score, q2.score, fill_color="#3B8686", line_color="black")

# whiskers (almost-0 height rects simpler than segments)
p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
p.rect(cats, upper.score, 0.2, 0.01, line_color="black")

# outliers
if not out.empty:
    p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = "white"
p.grid.grid_line_width = 2
p.xaxis.major_label_text_font_size="16px"

import numpy as np
from bokeh.plotting import figure, curdoc


sample_plot = figure(plot_height=400,
                     plot_width=400)
sample_plot.circle(x=np.random.normal(size=(10,)),
                   y=np.random.normal(size=(10,)))

output_file("Hoofdpagina.html", title="Hoofdpagina Dashboard V.A.")



#Creeer de kolommen voor de layout
Home = column(titel1, text1)
h1 = column(titel2, text2, data_table, text3, data_table2)
h2 = column(titel3, selectRegio, bar_chart)
h3 = column(titel4, text4, select, plot)


#Maak de tabs
# Create tab1 from plot p1: tab1
tab1 = Panel(child= Home , title='Hoofdpagina')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=h1, title='Hoofdstuk 1')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=h2, title='Hoofdstuk 2')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=h3, title='Hoofdstuk 3')

# Create tab4 from plot p4: tab4
tab5 = Panel(child=h2, title='Hoofdstuk 4')

# Create tab4 from plot p4: tab4
tab6 = Panel(child=h2, title='Bronvermelding')



# Create layout and add to current document

#layout = column(titel1, text1, titel2, text2, data_table, text3, data_table2, titel3, titel4, text4, select, plot)
layout = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5, tab6])
curdoc().add_root(layout)



