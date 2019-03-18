from bokeh.models.widgets import  Tabs
from bokeh.plotting import curdoc

import pandas as pd

import glob

import tab1_visualize
import tab2_analysis
import tab3_create_db
import tab4_learning
import tab5_testing
import tab6_classification
import tab7_deploy

data_all = glob.glob('./data/*.csv')

csv = pd.read_csv(data_all[0])

tab1 = tab1_visualize.tab_visualize(csv)
tab2 = tab2_analysis.tab_analysis(csv)
tab3 = tab3_create_db.tab_create_db(csv)
tab4 = tab4_learning.tab_learning()
tab5 = tab5_testing.tab_testing()
tab6 = tab6_classification.tab_classification()
tab7 = tab7_deploy.tab_deploy()

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6, tab7 ])
curdoc().add_root(tabs)


"""
tab7 = tab7_deploy.tab_deploy()

tabs = Tabs(tabs=[ tab7 ])
curdoc().add_root(tabs)
"""