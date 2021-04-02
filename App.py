# importeren libraries
from bokeh.io import curdoc
from bokeh.plotting import figure

# creeren van de 1-d plot

# maak de plot
plot = figure()

# voeg het type plot toe
plot.line([1,2,3,4,5], [2,5,4,6,7])

# Add the plot to the current document
curdoc().add_root(plot)
