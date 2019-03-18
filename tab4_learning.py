from bokeh.models.widgets import Panel
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, RangeSlider, Select, PreText, MultiSelect, RadioGroup, DataTable, TableColumn, Paragraph, TextInput
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import curdoc, figure, Figure
from bokeh.layouts import gridplot
from bokeh.models.glyphs import Circle
from bokeh.models import (BasicTicker, ColumnDataSource, Grid, LinearAxis, DataRange1d, PanTool, Plot, WheelZoomTool)
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
from bokeh.transform import transform
from bokeh.models.annotations import Title


import tensorflow as tf
from keras import backend as K

from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import os, glob, importlib

import pickle

import script.Classification as classification
import script.Regression as regression

def tab_learning():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    select_data = Select(title="Data:", value="", options=[])
    select_model = Select(title="Model script:", value="", options=[])

    data_descriptor = Paragraph(text=""" Data descriptor """, width=250, height=250)
    model_descriptor = Paragraph(text=""" Model descriptor """, width=250, height=250)

    select_grid = gridplot([[select_data, select_model], [data_descriptor, model_descriptor]])

    problem_type = RadioGroup(labels=["Classification", "Regression"], active=0)

    def problem_handler(new):
        if(new == 0):
            select_data.options = glob.glob('./np/Classification/*.npy')
            select_model.options = list(filter(lambda x: 'model_' in x, dir(classification)))

        elif(new == 1):
            select_data.options = glob.glob('./np/Regression/*.npy')
            select_model.options = list(filter(lambda x: 'model_' in x, dir(regression)))

    problem_type.on_click(problem_handler)

    learning_rate = TextInput(value="0.01", title="Learning rate")
    epoch_size = Slider(start=2, end=200, value=5, step=1, title="Epoch")
    batch_size = Slider(start=16, end=256, value=64, step=1, title="Batch")
    model_insert = TextInput(value="model", title="Model name")
    opeimizer = Select(title="Optimizer:", value="", options=["SGD", "ADAM", "RMS"])

    hyper_param = gridplot([[learning_rate], [epoch_size], [batch_size], [opeimizer], [model_insert]])

    xs = [[1], [1]]
    ys = [[1], [1]]
    label = [['Train loss'], ['Validation loss']]
    color = [['blue'], ['green']]
    total_loss_src = ColumnDataSource(data=dict(xs=xs, ys=ys, label=label, color=color))
    plot2 = Figure(plot_width=500, plot_height=300)
    plot2.multi_line('xs', 'ys', color='color', source=total_loss_src, line_width=3, line_alpha=0.6)
    TOOLTIPS = [("loss type", "@label"), ("loss value", "$y")]
    plot2.add_tools(HoverTool(tooltips=TOOLTIPS))
    t = Title()
    t.text = 'Loss'
    plot2.title = t

    acc_src = ColumnDataSource(data=dict(x=[1], y=[1], label=['R^2 score']))
    plot_acc = Figure(plot_width=500, plot_height=300, title="Accuracy")
    plot_acc.line('x', 'y', source=acc_src, line_width=3, line_alpha=0.7, color='red')
    TOOLTIPS = [("Type ", "@label"), ("Accuracy value", "$y")]
    plot_acc.add_tools(HoverTool(tooltips=TOOLTIPS))
    acc_list = []

    notifier = Paragraph(text=""" Notification """, width=200, height=100)

    def learning_handler():
        print("Start learning")
        del acc_list[:]

        tf.reset_default_graph()
        K.clear_session()

        data = np.load(select_data.value)
        data = data.item()

        print("data load complete")

        time_window = data.get('x').shape[-2]
        model_name = model_insert.value
        model_name = '(' + str(time_window) + ')' + model_name

        if (problem_type.active == 0):
            sub_path = 'Classification/'
        elif (problem_type.active == 1):
            sub_path = 'Regression/'

        model_save_dir = './model/' + sub_path + model_name + '/'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        x_shape = list(data.get('x').shape)
        print("Optimizer: " + str(opeimizer.value))

        print(select_model.value)

        if (problem_type.active == 0):
            target_model = getattr(classification, select_model.value)
            model = target_model(x_shape[-3], x_shape[-2], float(learning_rate.value), str(opeimizer.value),
                                 data.get('y').shape[-1])
        elif (problem_type.active == 1):
            target_model = getattr(regression, select_model.value)
            model = target_model(x_shape[-3], x_shape[-2], float(learning_rate.value), str(opeimizer.value),
                                 data.get('y').shape[-1])

        notifier.text = """ get model """

        training_epochs = int(epoch_size.value)
        batch = int(batch_size.value)
        loss_train = []
        loss_val = []

        train_ratio = 0.8
        train_x = data.get('x')
        train_y = data.get('y')
        length = train_x.shape[0]

        print(train_x.shape)

        data_descriptor.text = "Data shape: " + str(train_x.shape)
        # model_descriptor.text = "Model layer: " + str(model.model.summary())

        val_x = train_x[int(length * train_ratio):]
        if(val_x.shape[-1] == 1 and not 'cnn' in select_model.value):
            val_x = np.squeeze(val_x, -1)
        val_y = train_y[int(length * train_ratio):]

        train_x = train_x[:int(length * train_ratio)]
        if (train_x.shape[-1] == 1 and not 'cnn' in select_model.value):
            train_x = np.squeeze(train_x, -1)
        train_y = train_y[:int(length * train_ratio)]

        print(train_x.shape)

        if('model_dl' in select_model.value):
            for epoch in range(training_epochs):
                notifier.text = """ learning -- epoch: """ + str(epoch)

                hist = model.fit(train_x,
                                 train_y,
                                 epochs=1,
                                 batch_size=batch,
                                 validation_data=(val_x, val_y),
                                 verbose=1)

                print("%d epoch's cost:  %f" % (epoch, hist.history['loss'][0]))
                loss_train.append(hist.history['loss'][0])
                loss_val.append(hist.history['val_loss'][0])

                xs_temp = []
                xs_temp.append([i for i in range(epoch + 1)])
                xs_temp.append([i for i in range(epoch + 1)])

                ys_temp = []
                ys_temp.append(loss_train)
                ys_temp.append(loss_val)

                total_loss_src.data['xs'] = xs_temp
                total_loss_src.data['ys'] = ys_temp

                if (problem_type.active == 0):
                    r2 = hist.history['val_acc'][0]
                    label_str = 'Class accuracy'
                elif (problem_type.active == 1):
                    pred_y = model.predict(val_x)
                    r2 = r2_score(val_y, pred_y)
                    label_str = 'R^2 score'

                print("%d epoch's acc:  %f" % (epoch, r2))
                acc_list.append(np.max([r2, 0]))

                acc_src.data['x'] = [i for i in range(epoch+1)]
                acc_src.data['y'] = acc_list
                acc_src.data['label'] = [label_str for _ in range(epoch + 1)]

                print(acc_src.data)

            notifier.text = """ learning complete """
            model.save(model_save_dir + model_name + '.h5')
            notifier.text = """ model save complete """

            K.clear_session()

        elif('model_ml' in select_model.value):
            notifier.text = """ Machine learning model """

            if(train_x.shape[-2] != 1):
                notifier.text = """ Data include more then one time-frame. \n\n Data will automatically be flatten"""

            train_x = train_x.reshape([train_x.shape[0], -1])
            val_x = val_x.reshape([val_x.shape[0], -1])

            ##### shit
            if (problem_type.active == 0):
                train_y = np.argmax(train_y, axis=-1).astype(float)

            print(train_x.shape)
            print(train_y.shape)

            model.fit(train_x, train_y)
            notifier.text = """ Training done """
            pred_y = model.predict(val_x)

            print(pred_y)

            pickle.dump(model, open(model_save_dir + model_name + '.sav', 'wb'))
            notifier.text = """ Machine learning model saved """


    button_learning = Button(label="Run model")
    button_learning.on_click(learning_handler)

    learning_grid = gridplot(
        [[problem_type],
         [select_grid, hyper_param, button_learning, notifier],
         [plot2, plot_acc]])

    tab = Panel(child=learning_grid, title='Learning')

    return tab