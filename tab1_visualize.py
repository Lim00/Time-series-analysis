from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import WidgetBox, gridplot, Row, Column
from bokeh.models.widgets import Button, Slider, RangeSlider, Select, PreText, MultiSelect, RadioGroup, DataTable, \
    TableColumn, Paragraph
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import curdoc, figure

from bokeh.palettes import Category20_16

from sklearn import preprocessing
from sklearn import decomposition

import numpy as np
import pandas as pd


def tab_visualize(csv):

    csv_original = csv.copy()
    csv_modified = csv.copy()

    target_csv = {'csv': csv_original}

    source_length = target_csv['csv'].shape[0]

    table_source = ColumnDataSource(target_csv['csv'])
    table_columns = [TableColumn(field=col, title=col) for col in list(target_csv['csv'].columns)]

    table_original = DataTable(source=table_source, columns=table_columns, width=1550, height=250, fit_columns=False, name="Original")

    g = target_csv['csv'].columns.to_series().groupby(target_csv['csv'].dtypes).groups
    g_list = list(g.keys())

    float_list = []
    rest_list = []
    for l in g_list:
        if ('float' or 'int' in str(l)):
            float_list += list(g[l])
        else:
            rest_list += list(g[l])
    print(float_list)
    print(rest_list)

    def make_dataset(col_list, range_start, range_end):

        xs = []
        ys = []
        labels = []
        colors = []

        target_length = target_csv['csv'].shape[0]
        end = range_end if (range_end < target_length) else target_length

        target_data = target_csv['csv'][col_list + rest_list].iloc[range_start:end]
        target_x = list(np.arange(target_data.shape[0]) + range_start)

        for col, i in zip(col_list, range(len(col_list))):
            y = list(target_data[col].values)

            xs.append(target_x)
            ys.append(y)
            labels.append(col)
            colors.append(line_colors[i])

        new_src = ColumnDataSource(data={'x': xs, 'y': ys, 'label': labels, 'colors':colors})

        return new_src

    def make_plot(src):
        p = figure(plot_width=1300,
                   plot_height=400,
                   title='Raw data',
                   x_axis_label='Index',
                   y_axis_label='Sensor value')

        p.multi_line('x', 'y',
                     legend='label',
                     color='colors',
                     line_width=1,
                     source=src)

        tools = HoverTool()
        TOOLTIPS = [("", ""), ("", "")]

        return p

    def update(attr, old, new):

        column_to_plot = select_column.value

        new_src = make_dataset(column_to_plot, range_select.value[0], range_select.value[1])
        src.data.update(new_src.data)

    line_colors = Category20_16
    line_colors.sort()

    select_column = MultiSelect(title="Column visualization", value=float_list, options=float_list)
    select_column.on_change("value", update)

    range_select = RangeSlider(start=0, end=source_length, value=(0, int(source_length / 100)), step=1, title='Sensor range')
    range_select.on_change('value', update)

    src = make_dataset([float_list[0]], range_select.value[0], range_select.value[1])

    p = make_plot(src)

    select_normalize = Select(title="Select normalize transformation", options=["-","Min-Max", "Z normalize", "Raw"])
    button_get_normal = Button(label="Transform")

    def normal_handler():
        print("Normalize")
        # cols_to_normalize = float_list
        cols_to_normalize = select_transform.value

        if ( select_normalize.value == "Min-Max" ):
            scaler = preprocessing.MinMaxScaler()
            csv_modified[cols_to_normalize] = scaler.fit_transform(X=csv_original[cols_to_normalize].values)

            target_csv['csv'] = csv_modified
            table_source.data = ColumnDataSource(target_csv['csv']).data

            csv[cols_to_normalize] = target_csv['csv'][cols_to_normalize]
            print("Min Max Mormalize")

        elif ( select_normalize.value == "Z normalize" ):
            scaler = preprocessing.StandardScaler()
            csv_modified[cols_to_normalize] = scaler.fit_transform(X=csv_original[cols_to_normalize].values)

            target_csv['csv'] = csv_modified
            table_source.data = ColumnDataSource(target_csv['csv']).data

            csv[cols_to_normalize] = target_csv['csv'][cols_to_normalize]
            print("Z normalize")

        elif (select_normalize.value == "Raw"):

            csv_modified[cols_to_normalize] = csv_original[cols_to_normalize]

            target_csv['csv'] = csv_modified
            table_source.data = ColumnDataSource(target_csv['csv']).data

            csv[cols_to_normalize] = target_csv['csv'][cols_to_normalize]
            print("Raw")

    button_get_normal.on_click(normal_handler)

    select_transform = MultiSelect(title="Select columns for transformation", value=list(target_csv['csv'].columns), options=list(target_csv['csv'].columns))

    controls = WidgetBox(select_normalize, select_transform, button_get_normal, select_column, range_select)

    layout = Column(Row(table_original),
                    Row(controls, p))

    tab = Panel(child=layout, title='Visualization')

    return tab