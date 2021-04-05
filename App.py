# importeren libraries
from bokeh.io import curdoc
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import widgetbox, row, column
from bokeh.models.widgets import Panel, Tabs, Paragraph, Div, DataTable, DateFormatter, TableColumn
from bokeh.models import ColumnDataSource



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
titel3 = Div(text = "<h2"">Hoofdstuk 2: Histogrammen en boxplots""</h2>", width = 800, height = 50)
#h = figure(x_axis_label="Region", y_axis_label="GDP")
#h.vbar(x='Region', y='GDP', source=source)


# Hoofdstuk 3 Scatter en 2-D visualisatie
titel4 = Div(text = "<h2"">Hoofdstuk 3: Scatterplots en 2-D visualisaties""</h2>", width = 800, height = 50)
text4 = Div(text = "<h4"">Hieronder volgt scatterplot van prijs per ticket tegenover de GDP per capita""</h4>", width = 800, height = 50)

FareVsGDP = pd.read_csv('Data K\FareVsGDP.csv')

#scatterplot
s = figure(x_axis_label='GDP_Per_Capita', y_axis_label='Fare')
s.circle(FareVsGDP['GDP_Per_Capita'], FareVsGDP['Fare'], size=10)

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

bokeh_doc = curdoc()

sample_plot = figure(plot_height=400,
                     plot_width=400)
sample_plot.circle(x=np.random.normal(size=(10,)),
                   y=np.random.normal(size=(10,)))

output_file("Hoofdpagina.html", title="Hoofdpagina Dashboard V.A.")


dashboard = column(titel1, text1, titel2, text2, data_table, text3, data_table2, titel3, titel4, text4, s, sample_plot, p)
show(dashboard)



