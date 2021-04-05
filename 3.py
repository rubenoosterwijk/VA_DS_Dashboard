from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div, CheckboxGroup, CheckboxButtonGroup
from functools import partial



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
    0: dfdic["FirstClass"],
    1: dfdic['SecondClass'],
    2: dfdic['ThirdClass']
}

crewdictonary = {
    3: dfdic['DeckCrew'],
    4: dfdic['EngineeringCrew'],
    5: dfdic['Officers'],
    6: dfdic['OrchestraCrew'],
    7: dfdic['PostalCrew'],
    8: dfdic['RestaurantCrew'],
    9: dfdic['VictuallingCrew']
}

totaldictonary = {
    0: dfdic["FirstClass"],
    1: dfdic['SecondClass'],
    2: dfdic['ThirdClass'],
    3: dfdic['DeckCrew'],
    4: dfdic['EngineeringCrew'],
    5: dfdic['Officers'],
    6: dfdic['OrchestraCrew'],
    7: dfdic['PostalCrew'],
    8: dfdic['RestaurantCrew'],
    9: dfdic['VictuallingCrew']
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
    print(lijst)
    pass


def forwardData(list, data, lijst=[]):
    if list =='passengerlist':
        i = 0
        for a in data:
            print(a)
            lijst[i]= passengerdictonary[a]
            i = i +1

    if list == 'crewlist':
        i = 0
        for a in data:
            print(a)
            lijst[i] = crewdictonary[a]
            i = i + 1
    if list =='totallist':
        i = 0
        for a in data:
            print(a)
            lijst[i]= totaldictonary[a]
            i = i + 1

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
    forwardData("crewlist",  new)

def getBoth(attr, old, new):
    print(type(new))
    # new = [new]
    forwardData("totallist",  new)


d1.on_change('active', getPassengers)
d2.on_change('active', getCrew)
d3.on_change('active', getBoth)


curdoc().add_root(column(c, d1, d2, d3))
