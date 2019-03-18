from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, RangeSlider, Select, PreText, MultiSelect, RadioGroup, DataTable, TableColumn, Paragraph, TextInput

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import numpy as np
import pandas as pd

def tab_create_db(csv):

    csv_original = csv

    col_list = list(csv_original.columns)

    param_key = MultiSelect(title="Separator(Maximum 2)", options=col_list)
    param_x = MultiSelect(title="Predictor", options=col_list)
    param_y = Select(title="Response", options=col_list)

    problem_type = RadioGroup(labels=["Classification", "Regression"], active=0)

    notifier = Paragraph(text=""" DB """, width=200, height=100)

    def refresh_handler():
        col_list = list(csv_original.columns)

        param_key.options = col_list
        param_x.options = col_list
        param_y.options = col_list

    button_refresh = Button(label="Refresh column")
    button_refresh.on_click(refresh_handler)

    slider_window = Slider(start=1, end=30, value=15, step=1, title="Sliding window")
    slider_train_ratio = Slider(start=0, end=100, value=80, step=1, title="Train ratio")

    text_title = TextInput(value="", title="Title")

    button_create = Button(label="Create Numpy file")

    def create_handler():

        train_x = []
        train_y = []
        train_idx = []
        test_x = []
        test_y = []
        test_idx = []

        print(csv_original)

        if(problem_type.active == 1):
            notifier.text = """ Making DB - Regression """
        else:
            notifier.text = """ Making DB - Classification """

        if (len(param_key.value) == 0):

            slide = round(slider_window.value)

            xs = csv_original[param_x.value].values
            ys = csv_original[param_y.value].values

            train_ratio = round(slider_train_ratio.value)

            if(train_ratio != 0):

                train_x_set = xs[:xs.shape[0] * train_ratio / 100]
                train_y_set = ys[:ys.shape[0] * train_ratio / 100]

                for start, stop in zip(range(0, train_x_set.shape[0] - slide),
                                       range(slide, train_x_set.shape[0])):
                    train_x.append(train_x_set[start:start + slide])
                    train_y.append(train_y_set[start:start + slide][-1])
                    train_idx.append('0')

                train_x = np.asarray(train_x)
                train_x = np.swapaxes(train_x, 1, 2)
                train_x = np.expand_dims(train_x, -1)
                train_idx = np.asarray(train_idx)

            if(train_ratio != 100):

                test_x_set = xs[xs.shape[0] * train_ratio / 100:]
                test_y_set = ys[ys.shape[0] * train_ratio / 100:]

                for start, stop in zip(range(0, test_x_set.shape[0] - slide),
                                       range(slide, test_x_set.shape[0])):
                    test_x.append(test_x_set[start:start + slide])
                    test_y.append(test_y_set[start:start + slide][-1])
                    test_idx.append('1')

                test_x = np.asarray(test_x)
                test_x = np.swapaxes(test_x, 1, 2)
                test_x = np.expand_dims(test_x, -1)
                test_idx = np.asarray(test_idx)

            train_x = np.asarray(train_x)
            test_x = np.asarray(test_x)

            all_y = train_y + test_y

            data_train = {}
            data_test = {}

            if(problem_type.active == 1):
                train_y = np.asarray(train_y)
                train_y = np.expand_dims(train_y, -1)

                test_y = np.asarray(test_y)
                test_y = np.expand_dims(test_y, -1)

            elif(problem_type.active == 0):
                encoder = LabelEncoder()
                encoder.fit(all_y)
                encoded_y = encoder.transform(all_y)

                category_y = np_utils.to_categorical(encoded_y)

                labels = []
                for y in all_y:
                    if(y not in labels):
                        labels.append(y)

                data_train['labels'] = np.asarray(labels)
                data_test['labels'] = np.asarray(labels)

                train_y = category_y[:train_x.shape[0]]
                test_y = category_y[train_x.shape[0]:]


            data_train['x'] = train_x
            data_train['y'] = train_y
            data_train['key1'] = train_idx
            data_train['slideing_window'] = slider_window.value

            data_test['x'] = test_x
            data_test['y'] = test_y
            data_test['key1'] = test_idx
            data_test['slideing_window'] = slider_window.value

            if(problem_type.active == 1):
                print("Regression")
                target_dir = 'Regression/'
            elif (problem_type.active == 0):
                print("Classification")
                target_dir = 'Classification/'



            time_window = '[' + str(round(slider_window.value)) + ']'

            if (train_ratio != 0):
                np.save("./np/" + target_dir + time_window + text_title.value + "_train.npy", data_train)
            if (train_ratio != 100):
                np.save("./np/" + target_dir + time_window + text_title.value + "_test.npy", data_test)

        elif (len(param_key.value) == 1):

            key1_list = list(csv_original[param_key.value[0]].unique())

            train_ratio = round(slider_train_ratio.value)
            if(train_ratio == 0):
                train_key = []
                test_key = key1_list[int(len(key1_list) * train_ratio / 100):]
            elif(train_ratio == 100):
                train_key = key1_list[:int(len(key1_list) * train_ratio / 100)]
                test_key = []
            else:
                train_key = key1_list[:int(len(key1_list) * train_ratio / 100)]
                test_key = key1_list[int(len(key1_list) * train_ratio / 100):]

            for key in train_key:

                num_elements = csv_original[csv_original[param_key.value[0]] == key].shape[0]
                slide = round(slider_window.value)

                if (num_elements < slide):
                    continue

                xs = csv_original[csv_original[param_key.value[0]] == key][param_x.value].values
                ys = csv_original[csv_original[param_key.value[0]] == key][param_y.value].values

                for start, stop in zip(range(0, num_elements - slide),
                                       range(slide, num_elements)):
                    train_x.append(xs[start:start + slide])
                    train_y.append(ys[start:start + slide][-1])
                    train_idx.append(key)

            for key in test_key:
                num_elements = csv_original[csv_original[param_key.value[0]] == key].shape[0]
                slide = round(slider_window.value)

                if (num_elements < slide):
                    continue

                xs = csv_original[csv_original[param_key.value[0]] == key][param_x.value].values
                ys = csv_original[csv_original[param_key.value[0]] == key][param_y.value].values

                for start, stop in zip(range(0, num_elements - slide),
                                       range(slide, num_elements)):
                    test_x.append(xs[start:start + slide])
                    test_y.append(ys[start:start + slide][-1])
                    test_idx.append(key)


            all_y = train_y + test_y

            train_x = np.asarray(train_x)
            if (train_ratio != 0):
                train_x = np.swapaxes(train_x, 1, 2)
                train_x = np.expand_dims(train_x, -1)
            train_idx = np.asarray(train_idx)

            test_x = np.asarray(test_x)
            if (train_ratio != 100):
                test_x = np.swapaxes(test_x, 1, 2)
                test_x = np.expand_dims(test_x, -1)
            test_idx = np.asarray(test_idx)

            data_train = {}
            data_test = {}

            if(problem_type.active == 1):
                train_y = np.asarray(train_y)
                train_y = np.expand_dims(train_y, -1)

                test_y = np.asarray(test_y)
                test_y = np.expand_dims(test_y, -1)

            elif(problem_type.active == 0):
                encoder = LabelEncoder()
                encoder.fit(all_y)
                encoded_y = encoder.transform(all_y)

                category_y = np_utils.to_categorical(encoded_y)

                labels = []
                for y in all_y:
                    if(y not in labels):
                        labels.append(y)

                data_train['labels'] = np.asarray(labels)
                data_test['labels'] = np.asarray(labels)

                train_y = category_y[:train_x.shape[0]]
                test_y = category_y[train_x.shape[0]:]


            data_train['x'] = train_x
            data_train['y'] = train_y
            data_train['key1'] = train_idx
            data_train['slideing_window'] = slider_window.value

            data_test['x'] = test_x
            data_test['y'] = test_y
            data_test['key1'] = test_idx
            data_test['slideing_window'] = slider_window.value

            print(train_x.shape)
            print(train_y.shape)
            print(test_x.shape)
            print(test_y.shape)

            if(problem_type.active == 1):
                print("Regression")
                target_dir = 'Regression/'
            elif (problem_type.active == 0):
                print("Classification")
                target_dir = 'Classification/'

            print(train_x.shape)

            time_window = '[' + str(round(slider_window.value)) + ']'

            if (train_ratio == 0):
                np.save("./np/" + target_dir + time_window + text_title.value + "_test.npy", data_test)
            elif (train_ratio == 100):
                np.save("./np/" + target_dir + time_window + text_title.value + "_train.npy", data_train)
            else:
                np.save("./np/" + target_dir + time_window + text_title.value + "_train.npy", data_train)
                np.save("./np/" + target_dir + time_window + text_title.value + "_test.npy", data_test)


        elif (len(param_key.value) == 2):
            keys_list = csv_original[param_key.value].drop_duplicates()

            train_ratio = round(slider_train_ratio.value)
            if (train_ratio == 0):
                train_key = []
                test_key = keys_list.iloc[int(len(keys_list) * slider_train_ratio.value / 100):]
            elif (train_ratio == 100):
                train_key = keys_list.iloc[:int(len(keys_list) * slider_train_ratio.value / 100)]
                test_key = []
            else:
                train_key = keys_list.iloc[:int(len(keys_list) * slider_train_ratio.value / 100)]
                test_key = keys_list.iloc[int(len(keys_list) * slider_train_ratio.value / 100):]

            if (train_ratio != 0):
                for index, row in train_key.iterrows():

                    key1 = row[param_key.value[0]]
                    key2 = row[param_key.value[1]]

                    cond1 = csv_original[param_key.value[0]] == key1
                    cond2 = csv_original[param_key.value[0]] == key2

                    csv_target = csv_original[cond1 & cond2]

                    num_elements = csv_target.shape[0]
                    if (num_elements < slider_window.value):
                        continue

                    xs = csv_target[param_x.value].values
                    ys = csv_target[param_y.value].values

                    for start, stop in zip(range(0, num_elements - slider_window.value),
                                           range(slider_window.value, num_elements)):
                        train_x.append(xs[start:start + slider_window.value])
                        train_y.append(ys[start:start + slider_window.value][-1])
                        train_idx.append(str(key1) + "_" + str(key2))

            if (train_ratio != 100):
                for index, row in test_key.iterrows():

                    key1 = row[param_key.value[0]]
                    key2 = row[param_key.value[1]]

                    cond1 = csv_original[param_key.value[0]] == key1
                    cond2 = csv_original[param_key.value[1]] == key2

                    csv_target = csv_original[cond1 & cond2]

                    num_elements = csv_target.shape[0]
                    if (num_elements < slider_window.value):
                        continue

                    xs = csv_target[param_x.value].values
                    ys = csv_target[param_y.value].values

                    for start, stop in zip(range(0, num_elements - slider_window.value),
                                           range(slider_window.value, num_elements)):
                        test_x.append(xs[start:start + slider_window.value])
                        test_y.append(ys[start:start + slider_window.value][-1])
                        test_idx.append(str(key1) + "_" + str(key2))

            all_y = train_y + test_y

            train_x = np.asarray(train_x)
            if (train_ratio != 0):
                train_x = np.swapaxes(train_x, 1, 2)
                train_x = np.expand_dims(train_x, -1)
            train_idx = np.asarray(train_idx)

            test_x = np.asarray(test_x)
            if (train_ratio != 100):
                test_x = np.swapaxes(test_x, 1, 2)
                test_x = np.expand_dims(test_x, -1)
            test_idx = np.asarray(test_idx)

            data_train = {}
            data_test = {}

            if (problem_type.active == 1):
                train_y = np.asarray(train_y)
                train_y = np.expand_dims(train_y, -1)

                test_y = np.asarray(test_y)
                test_y = np.expand_dims(test_y, -1)

            elif (problem_type.active == 0):
                encoder = LabelEncoder()
                encoder.fit(all_y)
                encoded_y = encoder.transform(all_y)

                category_y = np_utils.to_categorical(encoded_y)

                labels = []
                for y in all_y:
                    if (y not in labels):
                        labels.append(y)

                data_train['labels'] = np.asarray(labels)
                data_test['labels'] = np.asarray(labels)

                train_y = category_y[:train_x.shape[0]]
                test_y = category_y[train_x.shape[0]:]

            data_train['x'] = train_x
            data_train['y'] = train_y
            data_train['key1'] = train_idx
            data_train['slideing_window'] = slider_window.value

            data_test['x'] = test_x
            data_test['y'] = test_y
            data_test['key1'] = test_idx
            data_test['slideing_window'] = slider_window.value

            print(train_x.shape)
            print(train_y.shape)
            print(test_x.shape)
            print(test_y.shape)

            if (problem_type.active == 1):
                print("Regression")
                target_dir = 'Regression/'
            elif (problem_type.active == 0):
                print("Classification")
                target_dir = 'Classification/'

            print(train_x.shape)

            time_window = '[' + str(round(slider_window.value)) + ']'

            if (train_ratio == 0):
                np.save("./np/" + target_dir + time_window + text_title.value + "_test.npy", data_test)
            elif (train_ratio == 100):
                np.save("./np/" + target_dir + time_window + text_title.value + "_train.npy", data_train)
            else:
                np.save("./np/" + target_dir + time_window + text_title.value + "_train.npy", data_train)
                np.save("./np/" + target_dir + time_window + text_title.value + "_test.npy", data_test)

        notifier.text = """ DB creation complete """

    button_create.on_click(create_handler)

    layout = Column(Row(param_key, param_x, param_y, notifier),
                    Row(problem_type),
                    Row(slider_window, slider_train_ratio, text_title),
                    Row(button_refresh, button_create))

    tab = Panel(child=layout, title='Create DB')

    return tab