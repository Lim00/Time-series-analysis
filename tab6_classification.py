from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, RangeSlider, Select, PreText, MultiSelect, RadioGroup, DataTable, TableColumn, Paragraph, TextInput
from bokeh.models import HoverTool, LinearColorMapper, Legend
from bokeh.plotting import curdoc, figure
from bokeh.models import (BasicTicker, ColumnDataSource, Grid, LinearAxis, DataRange1d, PanTool, Plot, WheelZoomTool)
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
from bokeh.palettes import PuBu
from bokeh.transform import transform

import pickle

import keras
import tensorflow as tf
from keras import backend as K

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
import os, time, glob

def tab_classification():

    data_list = glob.glob('./np/Classification/*.npy')
    data_list = sorted(data_list)
    select_data = Select(title="Data:", value="", options=data_list)

    model_list = os.listdir('./model/Classification/')
    model_list = sorted(model_list)
    select_model = Select(title="Trained Model", value="", options=model_list)

    notifier = Paragraph(text=""" Notification """, width=200, height=100)

    def refresh_handler():
        data_list_new = glob.glob('./np/Classification/*.npy')
        data_list_new = sorted(data_list_new)
        select_data.options = data_list_new

        model_list_new = os.listdir('./model/Classification/')
        model_list_new = sorted(model_list_new)
        select_model.options = model_list_new

    button_refresh = Button(label="Refresh list")
    button_refresh.on_click(refresh_handler)

    button_test = Button(label="Test model")

    #select_result = MultiSelect(title="Key(result):")

    colors = PuBu[9][:-1]
    colors.reverse()

    src_roc = ColumnDataSource(data=dict(xs=[], ys=[],line_color=[], label=[]))
    auc_paragraph = Paragraph(text="AUC")

    table_source = ColumnDataSource(pd.DataFrame())
    table_columns = [TableColumn(field=col, title=col) for col in ['-']]

    table_confusion = DataTable(source=table_source, columns=table_columns, width=800, height=200, fit_columns=True, name="Confusion matrix")

    def test_handler():
        tf.reset_default_graph()
        K.clear_session()

        DL = True

        print("Start test")
        tf.reset_default_graph()
        notifier.text = """ Start testing """

        if (select_data.value == ""):
            data = np.load(select_data.options[0])
        else:
            data = np.load(select_data.value)
        data = data.item()
        notifier.text = """ Import data """

        data_x = data.get('x')
        if (data_x.shape[-1] == 1 and not 'cnn' in select_model.value):
            data_x = np.squeeze(data_x, -1)
        data_y = data.get('y')

        print(data_x.shape)
        print(data_y.shape)
        print(data.get('key1'))

        if (select_model.value == ""):
            model_name = select_model.options[0]
        else:
            model_name = select_model.value
        model_save_dir = './model/Classification/' + model_name + '/'

        model_dl = glob.glob(model_save_dir + '*.h5')
        model_ml = glob.glob(model_save_dir + '*.sav')

        print(model_save_dir + model_name)

        if(len(model_dl) != 0 and len(model_ml) == 0):
            model = keras.models.load_model(model_save_dir + model_name + '.h5')
            DL =True

        elif(len(model_dl) == 0 and len(model_ml) != 0):
            model = pickle.load(open(model_save_dir + model_name + '.sav', 'rb'))

            data_x = data_x.reshape([data_x.shape[0], -1])
            data_y_ = np.copy(data_y)
            data_y = np.argmax(data_y, axis=-1).astype(float)

            print(data_x.shape)
            print(data_y.shape)
            DL = False

        notifier.text = """ Model restored """
        print("Model restored.")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        label = data.get('labels')
        print(label)

        if(DL):
            p_class = model.predict_classes(data_x)
            prob_class = model.predict(data_x)
            K.clear_session()

            true_class = [np.where(r == 1)[0][0] for r in data_y]
            true_class = np.asarray(true_class)

            x, y, _ = roc_curve(data_y.ravel(), prob_class.ravel())
            x = list(x)
            y = list(y)

            for i in range(data_y.shape[-1]):
                fp_, tp_, _ = roc_curve(data_y[:, i], prob_class[:, i])
                fpr[i] = list(fp_)
                tpr[i] = list(tp_)
                roc_auc[i] = auc(fpr[i], tpr[i])

            f1 = f1_score(true_class, p_class, average='macro')

        else:
            p_class = model.predict(data_x)
            prob_class = model.predict_proba(data_x)

            true_class = data_y

            x, y, _ = roc_curve(data_y_.ravel(), prob_class.ravel())
            x = list(x)
            y = list(y)

            for i in range(data_y_.shape[-1]):
                fp_, tp_, _ = roc_curve(data_y_[:, i], prob_class[:, i])
                fpr[i] = list(fp_)
                tpr[i] = list(tp_)
                roc_auc[i] = auc(fpr[i], tpr[i])

            print(fpr[0])
            print(type(fpr[0]))

            f1 = f1_score(true_class, p_class, average='macro')

        print(p_class.shape)
        print("Prediction complete")
        print(true_class.shape)

        cm_count = confusion_matrix(true_class, p_class)
        cm_ratio = cm_count.astype('float') / cm_count.sum(axis=1)[:, np.newaxis]

        def operate_on_Narray(A, B, function):
            try:
                return [operate_on_Narray(a, b, function) for a, b in zip(A, B)]
            except TypeError as e:
                # Not iterable
                return function(A, B)

        df = pd.DataFrame(cm_ratio, columns=label, index=label)

        tp = []
        tp_ = []
        fp = []

        for row in range(df.shape[0]):
            tp.append(format(df.iloc[row][row], '.2f'))
            tp_.append(df.iloc[row][row])
            fp.append(format(1 - df.iloc[row][row], '.2f'))

        df = pd.DataFrame(operate_on_Narray(cm_ratio, cm_count, lambda a, b: format(a, '.2f') + "(" + str(b) + ")"),
                          columns=label, index=label)

        df['True Positive'] = tp
        df['False Positive'] = fp

        df['Label'] = label
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        print(df)

        notifier.text = """ Calculation complete """

        table_confusion.columns = [TableColumn(field=col, title=col) for col in df.columns]
        table_source.data = ColumnDataSource(df).data
        table_confusion.source = table_source

        # AOC Curve

        xs = []
        ys = []
        line_color = []
        labels = []

        xs.append(x)
        ys.append(y)
        line_color.append('blue')
        labels.append("Macro ROC Curve")
        color_list = ['yellow', 'green', 'red', 'cyan', 'black']

        for key in fpr.keys():
            xs.append(fpr.get(key))
            ys.append(tpr.get(key))
            line_color.append(color_list[key % len(color_list)])
            labels.append("ROC Curve of class " + str(label[key]))

        roc_auc = auc(x, y)
        src_roc.data = ColumnDataSource(data=dict(xs=xs, ys=ys, line_color=line_color, label=labels)).data

        auc_paragraph.text = "Area under the curve: %f" % roc_auc

        history.text = history.text + "\n\t" + model_name + "'s accuracy / F1 score: " + format(np.mean(np.asarray(tp_)), '.2f') + ' / ' + format(f1, '.2f')

    button_test.on_click(test_handler)

    figure_roc = figure(title="ROC Curve", width=800, height=400)
    figure_roc.multi_line('xs', 'ys', line_color='line_color', legend='label', source=src_roc)
    history = PreText(text="", width=600, height=200)

    button_export = Button(label="Export result")

    def handler_export():
        df = table_source.data
        df = pd.DataFrame(df)

        print(df.columns)

        label_list = list(df['Label'])
        col_order = ['index', 'Label'] + label_list + ['True Positive', 'False Positive']
        df = df[col_order]

        # del df['index']

        df.to_csv('./Export/result.csv', index=False)

    button_export.on_click(handler_export)

    layout = Column(Row(button_refresh),
                    Row(select_data, select_model, button_test, notifier),
                    Row(table_confusion, history, button_export),
                    Row(Column(figure_roc, auc_paragraph)))

    tab = Panel(child=layout, title='Classification Test')

    return tab