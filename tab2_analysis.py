from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, Select, MultiSelect, RadioGroup, Dropdown , TableColumn, Paragraph, TextInput, CheckboxGroup, Div
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, WheelZoomTool

from bokeh.plotting import figure, Figure
from bokeh.models import (BasicTicker, ColumnDataSource, Grid, LinearAxis, DataRange1d, PanTool, Plot, WheelZoomTool)
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter, ContinuousColorMapper, FactorRange
from bokeh.transform import transform
from bokeh.palettes import Spectral6, RdYlBu, Greys, PRGn, RdBu
from bokeh.transform import linear_cmap

from sklearn import preprocessing
from sklearn import decomposition

import numpy as np
import pandas as pd
import math

import tab2_1_autoencoder as ae


def tab_analysis(csv):

    csv_original = csv

    g = csv_original.columns.to_series().groupby(csv_original.dtypes).groups
    g_list = list(g.keys())

    t = Figure()

    def convert(val, target):
        val_type = str(type(val))

        if ('float' in val_type):
            return float(target)
        elif ('int' in val_type):
            return int(target)
        elif ('str' in val_type):
            return str(target)

    box_figure = figure(tools="save", background_fill_color="#EFE8E2", title="Box", plot_width=500, plot_height=500, toolbar_location="below", x_range=[])
    box_figure.add_tools(WheelZoomTool())
    box_figure.add_tools(PanTool())

    corr_figure = figure(plot_width=500, plot_height=500, title="Correlation", toolbar_location=None, tools="", x_axis_location="above", x_range=[], y_range=[])

    def make_box_plot(df, param_list):
        df_box = pd.DataFrame(columns=['group', 'value'])

        for col in param_list:
            temp = pd.DataFrame(columns=['group', 'value'])
            temp['value'] = df[col].values
            temp['group'] = col

            df_box = pd.concat([df_box, temp])

        cats = param_list

        groups = df_box.groupby('group')
        q1 = groups.quantile(q=0.25)
        q2 = groups.quantile(q=0.5)
        q3 = groups.quantile(q=0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr

        # find the outliers for each category
        def outliers(group):
            cat = group.name
            return group[(group.value > upper.loc[cat]['value']) | (group.value < lower.loc[cat]['value'])]['value']

        out = groups.apply(outliers).dropna()

        # prepare outlier data for plotting, we need coordinates for every outlier.
        if not out.empty:
            outx = []
            outy = []
            for cat in cats:
                # only add outliers if they exist
                if not out.loc[cat].empty:
                    for value in out[cat]:
                        outx.append(cat)
                        outy.append(value)

        box_figure.x_range.factors = cats

        # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        qmin = groups.quantile(q=0.00)
        qmax = groups.quantile(q=1.00)
        upper.value = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, 'value']), upper.value)]
        lower.value = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, 'value']), lower.value)]

        # stems
        box_figure.segment(cats, upper.value, cats, q3.value, line_color="black")
        box_figure.segment(cats, lower.value, cats, q1.value, line_color="black")

        # boxes
        box_figure.vbar(cats, 0.7, q2.value, q3.value, fill_color="#E08E79", line_color="black")
        box_figure.vbar(cats, 0.7, q1.value, q2.value, fill_color="#3B8686", line_color="black")

        # whiskers (almost-0 height rects simpler than segments)
        box_figure.rect(cats, lower.value, 0.2, 0.01, line_color="black")
        box_figure.rect(cats, upper.value, 0.2, 0.01, line_color="black")

        # outliers
        if not out.empty:
            box_figure.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

        box_figure.xgrid.grid_line_color = None
        box_figure.ygrid.grid_line_color = "white"
        box_figure.grid.grid_line_width = 2
        box_figure.xaxis.major_label_text_font_size = "12pt"

    def make_correlation_plot(df, param_list):
        df_corr = df[param_list].corr().fillna(0)
        df_corr = df_corr.stack().rename("value").reset_index()

        print(df_corr)

        colors = RdBu[11]

        # Had a specific mapper to map color with value
        mapper = LinearColorMapper(palette=colors, low=-1, high=1)

        corr_figure.x_range.factors = list(df_corr.level_0.drop_duplicates())
        corr_figure.y_range.factors = list(df_corr.level_1.drop_duplicates())

        hover = HoverTool(tooltips=[("Corr", "@value"),])

        # Create rectangle for heatmap
        corr_figure.rect(x="level_0", y="level_1", width=1, height=1, source=ColumnDataSource(df_corr), line_color=None, fill_color=transform('value', mapper))
        corr_figure.add_tools(hover)

        # Add legend
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0))

        corr_figure.add_layout(color_bar, 'left')

    box_cor_x = MultiSelect(title="Predictor")
    button_box_corr = Button(label="Analysis response")

    def box_corr_handler():

        param_list = box_cor_x.value

        make_box_plot(csv_original, param_list)
        make_correlation_plot(csv_original, param_list)

    button_box_corr.on_click(box_corr_handler)


    param_key = MultiSelect(title="Separator(Maximum 2)")
    param_key.options = list(csv_original.columns)

    param_x = MultiSelect(title="Predictor")
    param_x.options = list(csv_original.columns)

    param_y = Select(title="Response")
    param_y.options = list(csv_original.columns)

    button_set = Button(label="Set parameter")


    key1 = MultiSelect(title="Key 1")
    key2 = MultiSelect(title="Key 2")

    target_x = Select(title="Sensor")
    #show_option = RadioGroup(labels=["Raw", "Moving average"], active=0)
    show_option = CheckboxGroup(labels=["Raw", "Moving average"], active=[0, 1])
    average_select = Slider(start=2, end=30, value=5, step=1, title='Average window')

    # 3rd row
    target_reduction = MultiSelect(title="Target for dimension reduction")
    reduction_method = Select(title="Dimension reduction", options=["PCA", "Autoencoder"])
    button_reduction = Button(label="Show result")
    figure_reduction = figure(tools="save, lasso_select", title="Dimension reduction result", plot_width=500, plot_height=500, toolbar_location="below")

    src = ColumnDataSource(data=dict(x=[], y=[], time=[]))
    # color_mapper = LinearColorMapper(palette='Viridis256', low=min(csv_original['group_index'].values), high=max(csv_original['group_index'].values))
    color_mapper = LinearColorMapper(palette='Viridis256', low=0, high=1000)
    figure_reduction.circle('x', 'y', source=src, size=5, color={'field': 'time', 'transform': color_mapper})
    TOOLTIPS = [("(x,y)", "($x, $y)"), ("Time", "@time")]
    figure_reduction.add_tools(HoverTool(tooltips=TOOLTIPS))

    src_reduction = ColumnDataSource(data=dict(center_x=[0], center_y=[0], radius=[0]))
    figure_reduction.circle("center_x", "center_y", radius="radius", source=src_reduction, alpha=0.3)


    def set_handler():

        if(len(param_key.value) >= 1):
            key1.options = list(map(lambda x: str(x), csv_original[param_key.value[0]].unique()))

        if (len(param_key.value) >= 2):
            key2.options = list(map(lambda x: str(x), csv_original[param_key.value[1]].unique()))

        if (len(param_key.value) != 0):
            csv_original['group_index'] = csv_original.groupby(param_key.value).cumcount() + 1  # Index per group -> consider it as time flow
        else:
            csv_original['group_index'] = [i + 1 for i in range(csv_original.shape[0])]

        x_list = []
        for col in param_x.value:
            if(csv_original[col].std() <= 0.0):
                continue

            x_list.append(col)

        target_x.options = x_list
        target_reduction.options = x_list
        box_cor_x.options = x_list

    button_set.on_click(set_handler)

    figure_multi_line = figure(tools="save", title="Sensor value per key", plot_width=1000, plot_height=500, toolbar_location="below")
    src1 = ColumnDataSource()
    button_sensor = Button(label="Show values")

    def sensor_hander():

        xs = []
        ys = []
        label_key = []
        colors = []
        line_width = []
        rolling_mean = int(average_select.value)

        if(target_x.value == ""):
            target_x.value = target_x.options[0]

        if (len(param_key.value) == 0):

            if (0 in show_option.active):

                y = csv_original[target_x.value].values
                x = np.arange(y.shape[0])
                xs.append(x)
                ys.append(y)
                label_key.append(str(target_x.value))
                colors.append(0)
                line_width.append(1)

            if (1 in show_option.active):

                y = csv_original[target_x.value].rolling(window=rolling_mean).mean().fillna(method='ffill').values
                x = np.arange(y.shape[0])
                xs.append(x)
                ys.append(y)
                label_key.append(str(target_x.value))
                colors.append(2)
                line_width.append(3)

        elif(len(param_key.value) == 1):
            cond1 = csv_original[param_key.value[0]].isin(key1.value)
            csv_slice = csv_original[cond1]

            for group in key1.value:
                if(0 in show_option.active):

                    print(csv_slice[param_key.value[0]].head())
                    print(csv_slice[param_key.value[0]].iloc[0])
                    print()

                    group_convert = convert(csv_slice[param_key.value[0]].iloc[0], group)

                    y = csv_slice[csv_slice[param_key.value[0]] == group_convert][target_x.value].values
                    x = np.arange(y.shape[0])
                    xs.append(x)
                    ys.append(y)
                    label_key.append(str(group_convert))
                    colors.append(group_convert)
                    line_width.append(1)

                if (1 in show_option.active):

                    group_convert = convert(csv_slice[param_key.value[0]].iloc[0], group)

                    y = csv_slice[csv_slice[param_key.value[0]] == group_convert][target_x.value].rolling(window=rolling_mean).mean().fillna(method='ffill').values
                    x = np.arange(y.shape[0])
                    xs.append(x)
                    ys.append(y)
                    label_key.append(str(group_convert))
                    colors.append(group_convert)
                    line_width.append(3)

        elif(len(param_key.value) == 2):
            cond1 = csv_original[param_key.value[0]].isin(key1.value)
            cond2 = csv_original[param_key.value[1]].isin(key2.value)
            csv_slice = csv_original[cond1 & cond2]

            # need type check

            for group1 in key1.value:
                for group2 in key2.value:
                    if (0 in show_option.active):

                        group_convert1 = convert(csv_slice[param_key.value[0]].iloc[0], group1)
                        group_convert2 = convert(csv_slice[param_key.value[1]].iloc[0], group2)

                        target_cond1 = csv_slice[param_key.value[0]] == group_convert1
                        target_cond2 = csv_slice[param_key.value[1]] == group_convert2

                        y = csv_slice[target_cond1 & target_cond2][target_x.value].values
                        x = np.arange(y.shape[0])
                        xs.append(x)
                        ys.append(y)
                        label_key.append(str(group_convert1) + " / " + str(group_convert2))
                        colors.append([group_convert1, group_convert2])
                        line_width.append(1)

                    if (1 in show_option.active):

                        group_convert1 = convert(csv_slice[param_key.value[0]].iloc[0], group1)
                        group_convert2 = convert(csv_slice[param_key.value[1]].iloc[0], group2)

                        target_cond1 = csv_slice[param_key.value[0]] == group_convert1
                        target_cond2 = csv_slice[param_key.value[1]] == group_convert2

                        y = csv_slice[target_cond1 & target_cond2][target_x.value].rolling(window=rolling_mean).mean().fillna(method='ffill').values
                        x = np.arange(y.shape[0])
                        xs.append(x)
                        ys.append(y)
                        label_key.append(str(group_convert1) + " / " + str(group_convert2))
                        colors.append([group_convert1, group_convert2])
                        line_width.append(3)

        color_all = [i for i in range(len(colors))]
        src1.data = ColumnDataSource(data=dict(xs=xs, ys=ys, label=label_key, color_all=color_all, line_width=line_width)).data

        figure_multi_line.multi_line('xs', 'ys', legend='label', source=src1, color=linear_cmap('color_all', "Viridis256", 0, len(colors)-1), line_width="line_width")
        ###
        TOOLTIPS = [("Keys", "@label"), ]
        figure_multi_line.add_tools(HoverTool(tooltips=TOOLTIPS))
        ###

    button_sensor.on_click(sensor_hander)

    def reduction_handler():

        print(reduction_method.options)
        print(reduction_method.value)

        x = csv_original[target_reduction.value].values

        if(reduction_method.value == "" or reduction_method.value == "PCA"):

            pca = decomposition.PCA(n_components=2)
            scaler = preprocessing.MinMaxScaler()

            result = pca.fit_transform(x)

            r_x = scaler.fit_transform(X=np.expand_dims(result[:, 0], -1))
            r_y = scaler.fit_transform(X=np.expand_dims(result[:, 1], -1))

            src.data = ColumnDataSource(data=dict(x=r_x, y=r_y, time=csv_original['group_index'].values)).data

            csv_original['reduction_1'] = r_x
            csv_original['reduction_2'] = r_y

            result_x_mean = np.mean(r_x)
            result_y_mean = np.mean(r_y)

            msg = "X center: " + str(result_x_mean) + "\n" + "_Y center: " + str(result_y_mean)

        else:
            print("Autoencoder")
            scaler = preprocessing.MinMaxScaler()

            result = ae.auto_encoder(x, 2, epoch=50)
            r_x = scaler.fit_transform(X=np.expand_dims(result[:, 0], -1))
            r_y = scaler.fit_transform(X=np.expand_dims(result[:, 1], -1))

            src.data = ColumnDataSource(data=dict(x=r_x, y=r_y, time=csv_original['group_index'].values)).data

            csv_original['reduction_1'] = r_x
            csv_original['reduction_2'] = r_y

            result_x_mean = np.mean(r_x)
            result_y_mean = np.mean(r_y)

            msg = "X center: " + str(result_x_mean) + "\n" + " Y_center: " + str(result_y_mean)

        color_mapper.low = min(csv_original['group_index'].values)
        color_mapper.high = max(csv_original['group_index'].values)
        src_reduction.data = ColumnDataSource(data=dict(center_x=[0], center_y=[0], radius=[0])).data
        ##################################################################################################################################### 컬럼 바꾸기
        csv_original['Classification_cutPoint'] = np.NaN
        csv_original['Classification_manualSelect'] = np.NaN

    button_reduction.on_click(reduction_handler)

    set_y_select = Dropdown(label="Select Y", menu=[("Linear", "item_1"), ("Weillibul", "item_2"), ("Piecewise", "item_3"), None, ("Dimension-reduction based select", "item_4")])
    new_y_setter_piecewise = TextInput(value="150", title="Piecewise cut point")
    new_y_setter_x = TextInput(value="", title="Regression center X")
    new_y_setter_y = TextInput(value="", title="Regression center Y")
    new_y_setter_r = TextInput(value="", title="Radius(out-of-bound value equal 0)")
    button_new_y = Button(label="Set new regression label")

    def set_y_handler():

        def y_linear(length):
            y = [i for i in range(length)]
            y.reverse()

            return y

        def y_weillibul(length, k=1.5, lmd=0.00002):
            dist_length = 1000
            y_ = [math.pow(math.e, -lmd * math.pow(i, k)) for i in range(dist_length + 1)]
            y_end = y_[dist_length]

            return [(y_[int((dist_length / length) * i)] - y_end) / (1.0 - y_end) * length for i in range(length)]

        def y_piecewise(length, const=130):

            y = [i for i in range(length)]
            y.reverse()
            y = [x if x < const else const for x in y]

            return y


        if(set_y_select.value == "item_1"):

            ys = []

            if(len(param_key.value) == 1):
                key1 = csv_original[param_key.value[0]].unique()
                for k in key1:
                    ys += y_linear(csv_original[csv_original[param_key.value[0]] == k].shape[0])

            elif (len(param_key.value) == 2):
                key1 = csv_original[param_key.value[0]].unique()
                for k in key1:
                    csv_temp = csv_original[csv_original[param_key.value[0]] == k]
                    key2 = csv_temp[param_key.value[1]].unique()

                    for k2 in key2:
                        ys += y_linear(csv_temp[csv_temp[param_key.value[1]] == k2].shape[0])

            csv_original['Regression_linear'] = ys

            msg = "Set Y value: Linear"
            message_y.text = msg

        if(set_y_select.value == "item_2"):

            ys = []

            if (len(param_key.value) == 1):
                key1 = csv_original[param_key.value[0]].unique()
                for k in key1:
                    ys += y_weillibul(csv_original[csv_original[param_key.value[0]] == k].shape[0])

            elif (len(param_key.value) == 2):
                key1 = csv_original[param_key.value[0]].unique()
                for k in key1:
                    csv_temp = csv_original[csv_original[param_key.value[0]] == k]
                    key2 = csv_temp[param_key.value[1]].unique()

                    for k2 in key2:
                        ys += y_weillibul(csv_temp[csv_temp[param_key.value[1]] == k2].shape[0])

            csv_original['Regression_weillibul'] = ys

            msg = "Set Y value: Weillibul"
            message_y.text = msg

        if (set_y_select.value == "item_3"):

            ys = []

            if (len(param_key.value) == 1):
                key1 = csv_original[param_key.value[0]].unique()
                for k in key1:
                    ys += y_piecewise(csv_original[csv_original[param_key.value[0]] == k].shape[0], int(new_y_setter_piecewise.value))

            elif (len(param_key.value) == 2):
                key1 = csv_original[param_key.value[0]].unique()
                for k in key1:
                    csv_temp = csv_original[csv_original[param_key.value[0]] == k]
                    key2 = csv_temp[param_key.value[1]].unique()

                    for k2 in key2:
                        ys += y_piecewise(csv_temp[csv_temp[param_key.value[1]] == k2].shape[0], int(new_y_setter_piecewise.value))

            csv_original['Regression_piecewise'] = ys

            msg = "Set Y value: Piecewise"
            message_y.text = msg

        if(set_y_select.value == "item_4"):
            center_x = float(new_y_setter_x.value)
            center_y = float(new_y_setter_y.value)
            radius = float(new_y_setter_r.value)

            """ add circle to the figure """
            # figure_reduction.circle([center_x], [center_y], radius=radius, alpha=0.3)

            src_reduction.data = ColumnDataSource(data=dict(center_x=[center_x], center_y=[center_y], radius=[radius])).data

            result_val = csv_original[['reduction_1', 'reduction_2']].values
            result_y = list(map(lambda xy: 1.0 - np.sqrt((center_x - xy[0]) ** 2 + (center_y - xy[1]) ** 2) / radius , result_val))
            result_y = [x if x >= 0.0 else 0 for x in result_y]

            csv_original['Regression_manualSelect'] = result_y

            #csv_original.to_csv('temp.csv')

            msg = "Set Y value: From reduction"
            message_y.text = msg

    button_new_y.on_click(set_y_handler)

    new_class_cut = TextInput(value="", title="Enter class cut point")
    new_class_setter = TextInput(value="", title="Enter class name")

    set_y_clss_select = Dropdown(label="Select Y",
                            menu=[("Manual cut", "item_1"), None, ("Dimension-reduction based select", "item_2") ])

    button_add_label = Button(label="Add classification label")
    button_new_class = Button(label="Set classification label")

    labels = []

    def label_adder_handler():
        indices = src.selected['1d']['indices']
        print(indices[:10])

        if(len(indices) == 0):
            label_notifier.text = "Non selected"
            return

        csv_original['Classification_manualSelect'].iloc[indices] = new_class_setter.value
        labels.append(new_class_setter.value)
        print("add label")
        label_notifier.text = str(labels)

    button_add_label.on_click(label_adder_handler)

    def label_all_handler():

        if(set_y_clss_select.value == "item_1"):
            class_cut = new_class_cut.value
            class_cut = list(map(int, class_cut.split(',')))

            # csv_original['Class'] = pd.np.digitize(csv_original['group_index'], bins=class_cut).astype(str)
            csv_original['Classification_cutPoint'] = pd.np.digitize(csv_original.groupby(param_key.value)['group_index'].transform(lambda x: x[::-1]), bins=class_cut).astype(str)

            label_notifier.text = "Labeling complete \n\nLabel: " + str(class_cut)
            del labels[:]

        elif (set_y_clss_select.value == "item_2"):
            if(csv_original['Classification_manualSelect'].isnull().any().any()):
                print("There is NaN values")

                csv_original['Classification_manualSelect'] = csv_original.fillna(method='ffill')['Classification_manualSelect'].values

            label_notifier.text = "Labeling complete \n\nLabel: " + str(labels)
            del labels[:]

        print(csv_original['Classification_cutPoint'])
        print(csv_original['Classification_manualSelect'])


    button_new_class.on_click(label_all_handler)

    head_regression = Div(text=""" <b>Set Regression Label</b> """)
    head_classification = Div(text=""" <b>Set Class Label</b> """)
    label_notifier = Paragraph(text=""" - """)


    button_export = Button(label="Export CSV")
    def handler_export():
        csv_original.to_csv('./Export/exported.csv', index=False)

    button_export.on_click(handler_export)

    message_y = Paragraph(text=""" - """, width=200, height=200)

    layout = Column(Row(param_key, param_x, param_y, button_set),
                    Row(Column(box_cor_x, button_box_corr), box_figure, corr_figure),
                    Row(Column(key1, key2), figure_multi_line, Column(target_x, show_option, average_select, button_sensor)),
                    Row(Column(reduction_method, target_reduction, button_reduction),
                        figure_reduction,
                        Column(head_regression, set_y_select, new_y_setter_piecewise, new_y_setter_x, new_y_setter_y, new_y_setter_r, button_new_y, message_y),
                        Column(head_classification, set_y_clss_select, new_class_cut, new_class_setter, button_add_label, button_new_class, label_notifier),
                        button_export))

    tab = Panel(child=layout, title='Analysis')

    return tab