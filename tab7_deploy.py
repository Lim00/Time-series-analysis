from bokeh.models.widgets import Panel, Tabs, Div
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, RangeSlider, Select, PreText, MultiSelect, RadioGroup, DataTable, \
    TableColumn, Paragraph
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import curdoc, figure, Figure

import tensorflow as tf

import keras
from keras import backend as K

from bokeh.palettes import Category20_16

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import pandas as pd

import glob, os, time


def tab_deploy():

    data_list = glob.glob('./Deploy/*.csv')
    data_list = sorted(data_list)
    select_data = Select(title="Stream source:", value="", options=data_list)

    model_list = os.listdir('./model/Regression/')
    model_list = sorted(model_list)
    select_model = Select(title="Model to deploy:", value="", options=model_list)

    src_health = ColumnDataSource(data=dict(x=[0], y=[0], color=['blue']))
    src_regressor = ColumnDataSource(data=dict(x=[0], y=[0]))
    src_warningBox = ColumnDataSource(data=dict(x=[0], y=[0]))
    src_alarmBox = ColumnDataSource(data=dict(x=[0], y=[0]))
    src_avg = ColumnDataSource(data=dict(x=[0], y=[0]))

    figure_health = figure(title="Health prediction result", width=800, height=460, y_range=(-0.1, 1.1))
    figure_health.circle(x='x', y='y', color='color', source=src_health, size=8)
    figure_health.line(x='x', y='y', source=src_regressor, color='red')
    figure_health.line(x='x', y='y', source=src_warningBox, color='green', alpha=0.5)
    figure_health.line(x='x', y='y', source=src_alarmBox, color='red', alpha=0.5)
    figure_health.line(x='x', y='y', source=src_avg, color='green', line_dash='dashed', line_width=10, alpha=0.5)


    # figure_health.line(x='x', y='y', source=src_health,  line_width=3, line_alpha=0.7)
    TOOLTIPS = [("(Time-stamp, Health index)", "($x, $y)")]
    figure_health.add_tools(HoverTool(tooltips=TOOLTIPS))

    src_trend1 = ColumnDataSource(data=dict(x=[0], y=[0]))
    src_trend2 = ColumnDataSource(data=dict(x=[0], y=[0]))

    figure_trend1 = figure(title="Sensor value trend 1", width=800, height=200)
    figure_trend1.circle(x='x', y='y', source=src_trend1, size=5)

    figure_trend2 = figure(title="Sensor value trend 2", width=800, height=200)
    figure_trend2.circle(x='x', y='y', source=src_trend2, size=5)

    rul_notifier = Div(text=""" <b> RUL based on Health Index </b> """)

    warning_level = 0.5
    alarm_level = 0.2

    def deploy_handler():

        src_regressor.data = ColumnDataSource(data=dict(x=[0], y=[0])).data
        src_avg.data = ColumnDataSource(data=dict(x=[0], y=[0])).data

        rul_notifier.text = """ <b> RUL estimate based on Health Index start </b> """

        def delay(c):
            time.sleep(c)

        print("Start deploy")

        tf.reset_default_graph()
        K.clear_session()

        if (select_model.value == ""):
            model_name = select_model.options[0]
        else:
            model_name = select_model.value
        model_save_dir = './model/Regression/' + model_name + '/'

        model = keras.models.load_model(model_save_dir + model_name + '.h5')

        slide = int(model_name.split(')')[0][1:])
        print(slide)
        print("Model load complete")

        if(select_data.value == ""):
            data_name = select_data.options[0]
        else:
            data_name = select_data.value

        df = pd.read_csv(data_name)

        print("data load complete")

        # Hard corded. Please do not mess with below code
        cols = ['s'+str(i+1) for i in range(21)]
        length = df.shape[0]


        x_idx = []
        y_value = []

        for start, stop in zip(range(0, length - slide),
                               range(slide, length)):
            print(start)

            data_patch = df.iloc[start:stop][cols].values
            data_patch = np.swapaxes(data_patch, 0, 1)
            data_patch = np.expand_dims(data_patch, 0)

            predict_point = model.predict(data_patch)[0][0]

            x_idx.append(float(start))
            y_value.append(predict_point)

        print("Calculation done")

        health_avg = []
        health_avg_x = []

        for i in range(len(x_idx)):
            src_health.data['x'] = np.asarray(x_idx)[:i]
            src_health.data['y'] = np.asarray(y_value)[:i]

            if(y_value[i] >= warning_level):
                src_health.data['color'] = ['blue' for _ in range(i + 1)]
            elif(y_value[i] >= alarm_level):
                src_health.data['color'] = ['red' for _ in range(i + 1)]
            else:
                src_health.data['color'] = ['black' for _ in range(i + 1)]

            src_warningBox.data = ColumnDataSource(data=dict(x=[0, i], y=[warning_level, warning_level])).data
            src_alarmBox.data = ColumnDataSource(data=dict(x=[0, i], y=[alarm_level, alarm_level])).data

            src_trend1.data['x'] = np.asarray(x_idx)[:i]
            src_trend1.data['y'] = df['s7'].iloc[0:i].values

            src_trend2.data['x'] = np.asarray(x_idx)[:i]
            src_trend2.data['y'] = df['s15'].iloc[0:i].values

            if(i > 50):
                reg_x = np.asarray(x_idx)[i-30:i]
                reg_x = np.expand_dims(reg_x, -1)
                reg_y = np.asarray(y_value)[i-30:i]

                regr = LinearRegression()
                reg = regr.fit(reg_x, reg_y)

                a = reg.coef_[0]
                b = reg.intercept_

                health_avg_x.append(np.asarray(x_idx)[i])
                health_avg.append(np.mean(np.asarray(y_value)[i-10:i]))

                src_avg.data['x'] = np.asarray(health_avg_x)
                src_avg.data['y'] = np.asarray(health_avg)

                if(reg.coef_[0] > 0 ):
                    rul_notifier.text = """ <h2> Stable </h2> """
                    delay(1.0)

                    continue

                if( (-b/a) - i > 180 ):
                    # rul_notifier.text = """ <strong> Deterioration begin (rate: %f) <strong> """ % (a)
                    rul_notifier.text = """ <h2> + 180 days left </h2> """

                else:
                    rul_notifier.text = """ <h2> <b> <ins> %d </ins> days left until breakdown </b> </h2> """ % (int((-b/a) - i))

                    src_regressor.data = ColumnDataSource(data=dict(x=[0, (-b/a)], y=[b, 0])).data
                    src_warningBox.data = ColumnDataSource(data=dict(x=[0, (-b/a)], y=[warning_level, warning_level])).data
                    src_alarmBox.data = ColumnDataSource(data=dict(x=[0, (-b/a)], y=[alarm_level, alarm_level])).data

            delay(1.0)

        rul_notifier.text = """ <h2> Streaming done </h2> """

    button_deploy = Button(label="Deploy")
    button_deploy.on_click(deploy_handler)

    layout = Column(Row(select_data, select_model),
                    button_deploy,
                    figure_trend1,
                    figure_trend2,
                    Row(figure_health, rul_notifier))

    tab = Panel(child=layout, title='Deploy')

    return tab