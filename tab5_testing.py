from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, RangeSlider, Select, PreText, MultiSelect, RadioGroup, DataTable, TableColumn, Paragraph, TextInput
from bokeh.models import HoverTool, LinearColorMapper
from bokeh.plotting import curdoc, figure
from bokeh.models import (BasicTicker, ColumnDataSource, Grid, LinearAxis, DataRange1d, PanTool, Plot, WheelZoomTool)
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter

import keras
import tensorflow as tf
from keras import backend as K
import pickle

from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import os, time, glob

def tab_testing():
    data_list = glob.glob('./np/Regression/*.npy')
    data_list = sorted(data_list)
    select_data = Select(title="Data:", value="", options=data_list)

    model_list = os.listdir('./model/Regression/')
    model_list = sorted(model_list)
    select_model = Select(title="Trained Model:", value="", options=model_list)

    notifier = Paragraph(text=""" Notification """, width=200, height=100)

    def refresh_handler():
        data_list_new = glob.glob('./np/Regression/*.npy')
        data_list_new = sorted(data_list_new)
        select_data.options = data_list_new

        model_list_new = os.listdir('./model/Regression/')
        model_list_new = sorted(model_list_new )
        select_model.options = model_list_new

    button_refresh = Button(label="Refresh list")
    button_refresh.on_click(refresh_handler)

    button_test = Button(label="Test model")

    select_result = MultiSelect(title="Key(result):")

    src = ColumnDataSource()
    df = pd.DataFrame(columns=['key', 'y', 'y_hat'])

    table_src = ColumnDataSource(pd.DataFrame(columns=['Key', 'MSE', 'R^2']))
    table_columns = [TableColumn(field=col, title=col) for col in ['Key', 'MSE', 'R^2']]
    table_acc = DataTable(source=table_src, columns=table_columns, width=350, height=400, fit_columns=True,
                                name="Accuracy per Key")


    def test_handler():
        df.drop(df.index, inplace=True)
        print("Start test")
        tf.reset_default_graph()
        K.clear_session()

        notifier.text = """ Start testing """

        if (select_data.value == ""):
            data = np.load(select_data.options[0])
        else:
            data = np.load(select_data.value)
        data = data.item()
        notifier.text = """ Import data """

        data_x = data.get('x')
        if(data_x.shape[-1] == 1 and not 'cnn' in select_model.value):
            data_x = np.squeeze(data_x, -1)
        data_y = data.get('y')
        data_key =  data.get('key1')

        print(data_x.shape)
        print(data_y.shape)

        df['key'] = data_key
        df['y'] = data_y[:, 0]

        op_list = []
        for i in df['key'].unique():
            op_list.append(str(i))
        select_result.options = op_list

        print(data_x.shape)
        print(data_y.shape)
        print(data.get('key1'))

        if (select_model.value == ""):
            model_name = select_model.options[0]
        else:
            model_name = select_model.value
        model_save_dir = './model/Regression/' + model_name + '/'

        model_dl = glob.glob(model_save_dir + '*.h5')
        model_ml = glob.glob(model_save_dir + '*.sav')

        print(model_save_dir + '*.h5')
        print(model_dl)
        print(model_ml)

        if (len(model_dl) > len(model_ml) ):
            model = keras.models.load_model(model_save_dir + model_name + '.h5')
            target_hat = model.predict(data_x)
            DL = True

        elif (len(model_dl) < len(model_ml) ):
            model = pickle.load(open(model_save_dir + model_name + '.sav', 'rb'))
            data_x = data_x.reshape([data_x.shape[0], -1])
            target_hat = model.predict(data_x)
            target_hat = np.expand_dims(target_hat, -1)
            DL = False

        notifier.text = """ Model restored """
        print("Model restored.")

        xs = []
        ys = []
        keys = []
        color = ['blue', 'red']

        xs.append([i for i in range(data_y.shape[0])])
        xs.append([i for i in range(data_y.shape[0])])
        ys.append(data_y)
        keys.append(data_key)

        print(target_hat.shape)
        K.clear_session()

        ys.append(target_hat)
        keys.append(data_key)

        print(target_hat[:, 0])
        df['y_hat'] = target_hat[:, 0]

        src.data = ColumnDataSource(data=dict(xs=xs, ys=ys, color=color, keys=keys)).data

        figure_trend.multi_line('xs', 'ys', source=src, color='color')

        line_mse = []
        line_r_2 = []
        for unit in df['key'].unique():
            target = df[df['key'] == unit]

            y = target['y'].values
            y_hat = target['y_hat'].values

            unit_mse = np.sum((y - y_hat) ** 2) / target.shape[0]
            unit_r_2 = np.max([r2_score(y, y_hat), 0])

            line_mse.append(unit_mse)
            line_r_2.append(unit_r_2)

        acc = pd.DataFrame(columns=['Key', 'MSE', 'R^2'])
        acc['Key'] = df['key'].unique()

        mse_mean = np.mean(line_mse)
        r_2_mean = np.mean(line_r_2)

        line_mse = list(map(lambda x: format(x, '.2f'), line_mse))
        acc['MSE'] = line_mse
        line_r_2 = list(map(lambda x: format(x, '.2f'), line_r_2))
        acc['R^2'] = line_r_2

        acc_append = pd.DataFrame(columns=acc.columns)
        acc_append['Key'] = ['MSE average', 'R^2 average']
        acc_append['MSE'] = [mse_mean, r_2_mean]

        acc = pd.concat([acc, acc_append])

        table_src.data = ColumnDataSource(acc).data

        notifier.text = """ Drawing complete """
        history.text = history.text + "\n\t" + model_name + "'s R^2 score: " + format(np.mean(r_2_mean), '.2f')


    def update(attr, old, new):

        key_to_plot = select_result.value

        xs = []
        ys = []
        keys = []

        y = []
        y_hat = []
        key = []

        key_type = type(df['key'].values[0])
        for k in key_to_plot:

            y += list(df[df['key'] == key_type(k)]['y'].values)
            y_hat += list(df[df['key'] == key_type(k)]['y_hat'].values)
            key += [k for _ in range(df[df['key'] == key_type(k)].shape[0])]

        ys.append(y)
        ys.append(y_hat)

        xs.append([i for i in range(len(y))])
        xs.append([i for i in range(len(y))])

        keys.append(key)
        keys.append(key)

        color = ['blue', 'red']

        src.data = ColumnDataSource(data=dict(xs=xs, ys=ys, color=color, keys=keys)).data

    select_result.on_change("value", update)

    button_test.on_click(test_handler)

    button_export = Button(label="Export result")

    def handler_export():
        df.to_csv('./Export/result.csv', index=False)
    button_export.on_click(handler_export)

    figure_trend = figure(title="Prediction result", width=800, height=460)
    history = PreText(text="", width=300, height=460)

    layout = Column(Row(button_refresh),
                    Row(select_data, select_model, button_test, select_result, notifier),
                    Row(table_acc, figure_trend, history, button_export))

    tab = Panel(child=layout, title='Regression Test')

    return tab