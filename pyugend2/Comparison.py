__author__ = 'krishnab'
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from operator import add, sub
from .ColumnSpecs import MODEL_RUN_COLUMNS, EXPORT_COLUMNS_FOR_CSV
from .PlotComposerOverallAttrition import PlotComposerOverallAttrition
from .PlotComposerOverallMFHiring import PlotComposerOverallMFHiring
from .PlotComposerAutoCorDeptSize import PlotComposerAutoCorDeptSize
from .PlotComposerLevelHiring import PlotComposerLevelHiring
import datetime


height = 700
width = 700
# CONSTANTS

# line_colors = ['#7fc97f', '#beaed4', '#fdc086','#386cb0','#f0027f','#ffff99']

class Comparison():
    def __init__(self, model_list):
        self.name = 'All Models'
        self.label = 'All Models'
        self.mlist = model_list

    def run_all_models(self, number_of_runs):
        for mod in self.mlist:
            mod.run_multiple(number_of_runs)

    def plot_comparison_overall_chart(self,
                           plottype,
                           intervals,
                           number_of_runs,
                           target,
                           xlabel,
                           ylabel,
                           title,
                           line_width=2,
                           width_=width,
                           height_=height,
                           transparency = 0.25,
                           linecolor=['blue', 'green', 'purple', 'brown'],
                           target_plot=False,
                           legend_location='top_right',
                           color_target='#ca0020',
                           percent_line_plot=False,
                           percent_line_value=0.5,
                           color_percent_line='#ca0020',
                           target_plot_linewidth=2,
                           percent_linewidth=2,
                           model_legend_label=['model'],
                           target_plot_legend_label='target',
                           percent_legend_label='percent',
                           male_female_numbers_plot=False,
                           mf_male_color=['#0004ff', '#2c7bb6'],
                           mf_target_color=['red', 'orange', 'deeppink','darkred'],
                           mf_male_label=['Male model 1', 'Male model 2'],
                           mf_target_label=['Target 1', 'Target 2'],
                           mf_male_linewidth=2,
                           mf_target_linewidth=2,
                           mf_data_color = ['blue'],
                           mf_data_label = ['Female Faculty Data', 'Male Faculty Data'],
                           mf_data_linewidth = [2],
                           parameter_sweep_param = None,
                           parameter_ubound = 0,
                           parameter_lbound = 0,
                           number_of_steps = 0,
                           vertical_line_label = 'Original Value',
                           vertical_line_width = 2,
                           vertical_line_color = ['black'],
                           data_plot = True,
                           data_line_color = ['blue'],
                           data_line_legend_label = 'Management Data',
                           year_offset = 0
                           ):


        # Choose plot type. This block will initialize the data for the
        # plots. If the plot is a parameter sweep, then it will run the
        # Models.run_parameter_sweep() function. If this is not a parameter
        # sweep, then the block will run the Models.run_multiple() function.



        # BEGIN BLOCK
        if plottype in ['parameter sweep percentage',
                        'parameter sweep probability']:

            for mod in self.mlist:
                mod.run_parameter_sweep(number_of_runs,
                                        parameter_sweep_param,
                                        parameter_lbound,
                                        parameter_ubound,
                                        number_of_steps)

            # xval, so I need to directly feed this range in.
            vertLineForSweepPlot = getattr(self.mlist[0], parameter_sweep_param)

            xval = self.mlist[0].parameter_sweep_results.loc[:,
                           'increment']

        else:

            for mod in self.mlist:
                mod.run_multiple(number_of_runs)

            xval = list(range(min([m.duration for m in self.mlist])))[year_offset:]

        # END OF BLOCK



        # BEGIN BLOCK
        # This section  just initializes the bokeh figure with labels and
        # sizing.
        p = figure(title=title,
               x_axis_label=xlabel,
               y_axis_label=ylabel,
               width=width_,
               height=height_)

        # END BLOCK

        # BEGIN BLOCK
        # Generate main plot values.
        # This section will take the data from the model simulation and
        # convert it to the appropriate y-variables for the plot.


        if plottype == 'probability proportion':

            for mod in self.mlist:
                mod.run_probability_analysis_gender_proportion(number_of_runs,
                                                            target)

            yval = [m.probability_matrix['Probability'] for m in self.mlist]

            data_plot = False

        if plottype == 'gender proportion':

            yval = [m.results_matrix['mean_gendprop'] for m in self.mlist]

            dval = self.mlist[0].mgmt_data.get_field('gender_proportion_dept')

        if plottype == 'gender numbers':
            yval = [sum(list([m.results_matrix['mean_f1'],
                              m.results_matrix['mean_f2'],
                              m.results_matrix['mean_f3']])) for m in self.mlist]

            total_faculty = [sum(list([m.results_matrix['mean_m1'],
                                       m.results_matrix['mean_m2'],
                                       m.results_matrix['mean_m3'],
                                       m.results_matrix['mean_f1'],
                                       m.results_matrix['mean_f2'],
                                       m.results_matrix['mean_f3']])) for m in self.mlist]

            number_dynamic_target = [np.round(target * dept) for dept in total_faculty]

            dval = self.mlist[0].mgmt_data.get_field('total_females_dept')


            pass

        if plottype == 'unfilled vacancies':
            yval = [m.results_matrix['mean_unfilled'] for m in self.mlist]

            dval = self.mlist[0].mgmt_data.get_field('excess_hires')

        if plottype == 'department size':
            yval = [m.results_matrix['mean_dept_size'] for m in self.mlist]

            dval = self.mlist[0].mgmt_data.get_field('department_size')

        if plottype == 'male female numbers':

            yval = [sum(list([m.results_matrix['mean_f1'],
                             m.results_matrix['mean_f2'],
                             m.results_matrix['mean_f3']])) for m in self.mlist]



            yval2 = [sum(list([m.results_matrix['mean_m1'],
                             m.results_matrix['mean_m2'],
                             m.results_matrix['mean_m3']])) for m in self.mlist]


            total_faculty = [sum(list([m.results_matrix['mean_m1'],
                             m.results_matrix['mean_m2'],
                             m.results_matrix['mean_m3'],
                             m.results_matrix['mean_f1'],
                             m.results_matrix['mean_f2'],
                             m.results_matrix['mean_f3']])) for m in self.mlist]

            yval3 = [np.round(target * dept) for dept in total_faculty]

            dval_male = self.mlist[0].mgmt_data.get_field('total_males_dept')
            dval_female = self.mlist[0].mgmt_data.get_field('total_females_dept')

        if plottype == 'parameter sweep percentage':

            yval = [m.parameter_sweep_results['mean_gendprop'] for m in
                    self.mlist]

            data_plot = False

        if plottype == 'parameter sweep probability':

            for mod in self.mlist:
                mod.run_probability_parameter_sweep_overall(number_of_runs,
                                                            parameter_sweep_param,
                                                            parameter_lbound,
                                                            parameter_ubound,
                                                            number_of_steps,
                                                            target)

            yval = [m.probability_sweep_results['Probability'] for m in
                    self.mlist]


            data_plot = False
        # Set confidence bounds using empirical results

        if intervals == 'empirical':
            if plottype == 'probability proportion':

                upper_band = yval
                lower_band = yval


            if plottype == 'gender proportion':

                upper_band = [m.results_matrix['gendprop_975'] for m in
                              self.mlist]
                lower_band = [m.results_matrix['gendprop_025'] for m in
                              self.mlist]

            if plottype == 'gender numbers':
                upper_band = [sum(list([m.results_matrix['f1_975'],
                             m.results_matrix['f2_975'],
                             m.results_matrix['f3_975']])) for m in self.mlist]

                lower_band = [sum(list([m.results_matrix['f1_025'],
                             m.results_matrix['f2_025'],
                             m.results_matrix['f3_025']])) for m in self.mlist]

            if plottype == 'unfilled vacancies':
                upper_band = [m.results_matrix['unfilled_975'] for m in
                              self.mlist]
                lower_band = [m.results_matrix['unfilled_025'] for m in
                              self.mlist]

            if plottype == 'department size':

                upper_band = [m.results_matrix['dept_size_975'] for m in
                              self.mlist]
                lower_band = [m.results_matrix['dept_size_025'] for m in
                              self.mlist]

            if plottype == 'male female numbers':
                upper_band = yval
                lower_band = yval

            if plottype == 'parameter sweep percentage':
                upper_band = [m.parameter_sweep_results['gendprop_975'] for m in
                              self.mlist]
                lower_band = [m.parameter_sweep_results['gendprop_025'] for m in
                              self.mlist]

            if plottype == 'parameter sweep probability':
                upper_band = yval
                lower_band = yval


        # Set confidence bounds using 2 standard deviations

        if intervals == 'standard':

            if plottype == 'probability proportion':

                upper_band = yval
                lower_band = yval

            if plottype == 'gender proportion':

                fill_matrix = [m.results_matrix['std_gendprop'] for m in
                               self.mlist]

                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'gender numbers':
                pass

            if plottype == 'unfilled vacancies':
                fill_matrix = [m.results_matrix['std_unfilled'] for m in
                               self.mlist]
                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))
                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'department size':
                fill_matrix = [m.results_matrix['std_dept_size'] for m in
                               self.mlist]
                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))
                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'male female numbers':

                upper_band = yval
                lower_band = yval


            if plottype == 'parameter sweep percentage':

                fill_matrix = [m.parameter_sweep_results['std_gendprop'] for m in
                               self.mlist]

                upper_band = list(map(add, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

                lower_band = list(map(sub, [y for y in yval],[1.96*f for f in
                                                         fill_matrix]))

            if plottype == 'parameter sweep probability':
                upper_band = yval
                lower_band = yval

        # Execute plots

        for k, v in enumerate(self.mlist):

            p.line(xval, yval[k],
                   line_width=line_width,
                   line_color=linecolor[k],
                   legend=model_legend_label[k])

            p.circle(xval, yval[k],
                     color=linecolor[k],
                     size=3)

            x_data = np.asarray(xval)
            band_x = np.append(x_data, x_data[::-1])
            band_y = np.append(lower_band[k], upper_band[k][::-1])

            p.patch(band_x,
                    band_y,
                    color=linecolor[k],
                    fill_alpha=transparency)

            if plottype in ['parameter sweep percentage',
                            'parameter sweep probability']:
                p.line(vertLineForSweepPlot, np.linspace(0, max(band_y)),
                       line_width = vertical_line_width,
                       line_color = vertical_line_color[k],
                       legend = vertical_line_label,
                       line_dash=[6,6])

            if male_female_numbers_plot:
                p.line(xval,
                       yval2[k],
                       line_color=mf_male_color[k],
                       legend=mf_male_label[k],
                       line_width=mf_male_linewidth)

                p.circle(xval, yval2[k],
                         color=mf_male_color[k],
                         size=3)

                p.line(xval,
                       yval3[k],
                       line_color=mf_target_color[k],
                       legend=mf_target_label[k],
                       line_width=mf_target_linewidth,
                       line_dash=[6,6])


        if target_plot:
            p.line(xval, target,
                   line_color=color_target,
                   legend=target_plot_legend_label,
                   line_width=target_plot_linewidth,
                   line_dash=[6, 6])

        if percent_line_plot:
            p.line(xval, percent_line_value,
                   line_color=color_percent_line,
                   legend=percent_legend_label,
                   line_width=percent_linewidth,
                   line_dash=[2, 2])

        if data_plot and plottype != 'male female numbers':
            p.line(xval, dval,
                    line_color = data_line_color[0],
                    legend = data_line_legend_label,
                    line_width = line_width,
                    line_dash = 'dashdot')

        if data_plot and plottype == 'male female numbers':
            p.line(xval,
                   dval_female,
                   line_color=mf_data_color[0],
                   legend=mf_data_label[0],
                   line_width=mf_data_linewidth[0],
                   line_dash=[4, 4])

            p.line(xval,
                   dval_male,
                   line_color=mf_data_color[0],
                   legend=mf_data_label[1],
                   line_width=mf_data_linewidth[0],
                   line_dash=[4,4])

        if plottype == 'gender numbers':
            p.line(xval,
                   number_dynamic_target[k],
                   line_color=color_target,
                   legend=target_plot_legend_label,
                   line_width=target_plot_linewidth,
                   line_dash=[6, 6])


        return(p)


    def plot_comparison_level_chart(self,
                         plottype,
                         intervals,
                         number_of_runs,
                         target,
                         xlabels,
                         ylabels,
                         titles = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3'],
                         line_width=2,
                         height_= height // 2,
                         width_ = width // 2,
                         legend_location='top right',
                         model_legend_label=['model1', 'model2'],
                         transparency = 0.25,
                         linecolor=['green', 'purple', 'blue', 'brown'],
                         target_plot=False,
                         target_color='#ca0020',
                         target_number_color = ['red', 'orange','deeppink','darkred' ],
                         target_number_labels = ['Target, Model 1',
                                               'Target, Model 2',
                                               'Target, Model 3'],
                         target_plot_linewidth=2,
                         target_plot_legend_label='target',
                         percent_line_plot=False,
                         percent_line_value=0.5,
                         color_percent_line='#ca0020',
                         percent_linewidth=2,
                         percent_legend_label='percent',
                         parameter_sweep_param=None,
                         parameter_ubound=0,
                         parameter_lbound=0,
                         number_of_steps=0,
                         vertical_line_label='Original Value',
                         vertical_line_width = 2,
                         vertical_line_color = ['black', 'purple'],
                         data_plot=True,
                         data_line_color=['blue'],
                         data_line_legend_label=['Management Data']
                         ):

        # Choose plot type. This block will initialize the data for the
        # plots. If the plot is a parameter sweep, then it will run the
        # Models.run_parameter_sweep() function. If this is not a parameter
        # sweep, then the block will run the Models.run_multiple() function.

        # BEGIN BLOCK

        if plottype in ['parameter sweep percentage',
                        'parameter sweep probability',
                        'parameter sweep number']:

            for mod in self.mlist:
                mod.run_parameter_sweep(number_of_runs,
                                        parameter_sweep_param,
                                        parameter_lbound,
                                        parameter_ubound,
                                        number_of_steps)

            # xval, so I need to directly feed this range in.

            vertLineForSweepPlot = getattr(self.mlist[0], parameter_sweep_param)
            xval = self.mlist[0].parameter_sweep_results.loc[:, 'increment']

        else:

            for mod in self.mlist:
                mod.run_multiple(number_of_runs)

            xval = list(range(min([m.duration for m in self.mlist])))

        # END BLOCK

        # Generate main plot values.
        # This section will take the data from the model simulation and
        # convert it to the appropriate y-variables for the plot.
        # BEGIN BLOCK

        dval_f1 = 0
        dval_f2 = 0
        dval_f3 = 0
        dval_m1 = 0
        dval_m2 = 0
        dval_m3 = 0

        if plottype == 'probability proportion':
            for mod in self.mlist:
                mod.run_probability_analysis_gender_by_level(number_of_runs,
                                                             target)

            # Not a very finesse way to do these assignments, but it makes
            # the code more readable.
            yval_f1 = [m.probability_by_level['pf1'] for m in self.mlist]
            yval_f2 = [m.probability_by_level['pf2'] for m in self.mlist]
            yval_f3 = [m.probability_by_level['pf3'] for m in self.mlist]
            yval_m1 = [m.probability_by_level['pm1'] for m in self.mlist]
            yval_m2 = [m.probability_by_level['pm2'] for m in self.mlist]
            yval_m3 = [m.probability_by_level['pm3'] for m in self.mlist]

            data_plot = False

        if plottype == 'gender proportion':

            female_pct_matrices = [m.pct_female_matrix for m in self.mlist]

            yval_f1 = [m['mpct_f1'] for m in female_pct_matrices]
            yval_f2 = [m['mpct_f2'] for m in female_pct_matrices]
            yval_f3 = [m['mpct_f3'] for m in female_pct_matrices]
            yval_m1 = [m['mpct_m1'] for m in female_pct_matrices]
            yval_m2 = [m['mpct_m2'] for m in female_pct_matrices]
            yval_m3 = [m['mpct_m3'] for m in female_pct_matrices]

            dval_f1 = self.mlist[0].mgmt_data.get_field('gender_proportion_1')
            dval_f2 = self.mlist[0].mgmt_data.get_field('gender_proportion_2')
            dval_f3 = self.mlist[0].mgmt_data.get_field('gender_proportion_3')
            dval_m1 = [1 - x for x in self.mlist[0].mgmt_data.get_field('gender_proportion_1')]
            dval_m2 = [1 - x for x in self.mlist[0].mgmt_data.get_field('gender_proportion_2')]
            dval_m3 = [1 - x for x in self.mlist[0].mgmt_data.get_field('gender_proportion_3')]

        if plottype == 'gender number':

            mean_matrices = [m.results_matrix.loc[:,'mean_f1':'mean_m3'] for
                             m in self.mlist]
            yval_f1 = [m['mean_f1'] for m in mean_matrices]
            yval_f2 = [m['mean_f2'] for m in mean_matrices]
            yval_f3 = [m['mean_f3'] for m in mean_matrices]
            yval_m1 = [m['mean_m1'] for m in mean_matrices]
            yval_m2 = [m['mean_m2'] for m in mean_matrices]
            yval_m3 = [m['mean_m3'] for m in mean_matrices]

            dval_f1 = self.mlist[0].mgmt_data.get_field('number_f1')
            dval_f2 = self.mlist[0].mgmt_data.get_field('number_f2')
            dval_f3 = self.mlist[0].mgmt_data.get_field('number_f3')
            dval_m1 = self.mlist[0].mgmt_data.get_field('number_m1')
            dval_m2 = self.mlist[0].mgmt_data.get_field('number_m2')
            dval_m3 = self.mlist[0].mgmt_data.get_field('number_m3')

            total_1 = [sum(list([m.results_matrix['mean_f1'],
                                   m.results_matrix['mean_m1']])) for m in
                         self.mlist]
            total_2 = [sum(list([m.results_matrix['mean_f2'],
                                   m.results_matrix['mean_m2']])) for m in
                         self.mlist]
            total_3 = [sum(list([m.results_matrix['mean_f3'],
                               m.results_matrix['mean_m3']])) for m in
                        self.mlist]
            target_f1 = [np.round(target * level) for level in total_1]
            target_f2 = [np.round(target * level) for level in total_2]
            target_f3 = [np.round(target * level) for level in total_3]
            target_m1 = [np.round((1-target) * level) for level in total_1]
            target_m2 = [np.round((1-target) * level) for level in total_2]
            target_m3 = [np.round((1-target) * level) for level in total_3]

            num_targets = [target_f1,
                           target_f2,
                           target_f3,
                           target_m1,
                           target_m2,
                           target_m3]
        if plottype == 'parameter sweep percentage':

            female_sweep_matrices = [m.parameter_sweep_results for m in
                                   self.mlist]

            yval_f1 = [m['mpct_f1'] for m in female_sweep_matrices]
            yval_f2 = [m['mpct_f2'] for m in female_sweep_matrices]
            yval_f3 = [m['mpct_f3'] for m in female_sweep_matrices]
            yval_m1 = [m['mpct_m1'] for m in female_sweep_matrices]
            yval_m2 = [m['mpct_m2'] for m in female_sweep_matrices]
            yval_m3 = [m['mpct_m3'] for m in female_sweep_matrices]

            data_plot = False
        if plottype == 'parameter sweep number':

            female_sweep_matrices = [m.parameter_sweep_results for m in
                                   self.mlist]

            yval_f1 = [m['mean_f1'] for m in female_sweep_matrices]
            yval_f2 = [m['mean_f2'] for m in female_sweep_matrices]
            yval_f3 = [m['mean_f3'] for m in female_sweep_matrices]
            yval_m1 = [m['mean_m1'] for m in female_sweep_matrices]
            yval_m2 = [m['mean_m2'] for m in female_sweep_matrices]
            yval_m3 = [m['mean_m3'] for m in female_sweep_matrices]

            data_plot = False

        if plottype == 'parameter sweep probability':
            data_plot = False
            pass

        #END BLOCK

        # setup empirical bounds

        if intervals == 'empirical':

            if plottype == 'probability proportion':

                upper_f1 = yval_f1
                upper_f2 = yval_f2
                upper_f3 = yval_f3
                upper_m1 = yval_m1
                upper_m2 = yval_m2
                upper_m3 = yval_m3

                lower_f1 = yval_f1
                lower_f2 = yval_f2
                lower_f3 = yval_f3
                lower_m1 = yval_m1
                lower_m2 = yval_m2
                lower_m3 = yval_m3

            if plottype == 'gender proportion':

                upper_f1 = [m['pf1_975'] for m in female_pct_matrices]
                upper_f2 = [m['pf2_975'] for m in female_pct_matrices]
                upper_f3 = [m['pf3_975'] for m in female_pct_matrices]
                upper_m1 = [m['pm1_975'] for m in female_pct_matrices]
                upper_m2 = [m['pm2_975'] for m in female_pct_matrices]
                upper_m3 = [m['pm3_975'] for m in female_pct_matrices]

                # print('band', upper_f2)
                # print('\n')
                lower_f1 = [m['pf1_025'] for m in female_pct_matrices]
                lower_f2 = [m['pf2_025'] for m in female_pct_matrices]
                lower_f3 = [m['pf3_025'] for m in female_pct_matrices]
                lower_m1 = [m['pm1_025'] for m in female_pct_matrices]
                lower_m2 = [m['pm2_025'] for m in female_pct_matrices]
                lower_m3 = [m['pm3_025'] for m in female_pct_matrices]


            if plottype == 'gender number':
                u_matrices = [m.results_matrix.loc[:, 'f1_975':'m3_975'] for
                              m in
                            self.mlist]

                upper_f1 = [m['f1_975'] for m in u_matrices]
                upper_f2 = [m['f2_975'] for m in u_matrices]
                upper_f3 = [m['f3_975'] for m in u_matrices]
                upper_m1 = [m['m1_975'] for m in u_matrices]
                upper_m2 = [m['m2_975'] for m in u_matrices]
                upper_m3 = [m['m3_975'] for m in u_matrices]

                l_matrices = [m.results_matrix.loc[:, 'f1_025':'m3_025'] for
                              m in
                            self.mlist]

                lower_f1 = [m['f1_025'] for m in l_matrices]
                lower_f2 = [m['f2_025'] for m in l_matrices]
                lower_f3 = [m['f3_025'] for m in l_matrices]
                lower_m1 = [m['m1_025'] for m in l_matrices]
                lower_m2 = [m['m2_025'] for m in l_matrices]
                lower_m3 = [m['m3_025'] for m in l_matrices]

            if plottype == 'parameter sweep percentage':

                upper_f1 = [m['pf1_975'] for m in female_sweep_matrices]
                upper_f2 = [m['pf2_975'] for m in female_sweep_matrices]
                upper_f3 = [m['pf3_975'] for m in female_sweep_matrices]
                upper_m1 = [m['pm1_975'] for m in female_sweep_matrices]
                upper_m2 = [m['pm2_975'] for m in female_sweep_matrices]
                upper_m3 = [m['pm3_975'] for m in female_sweep_matrices]

                lower_f1 = [m['pf1_025'] for m in female_sweep_matrices]
                lower_f2 = [m['pf2_025'] for m in female_sweep_matrices]
                lower_f3 = [m['pf3_025'] for m in female_sweep_matrices]
                lower_m1 = [m['pm1_025'] for m in female_sweep_matrices]
                lower_m2 = [m['pm2_025'] for m in female_sweep_matrices]
                lower_m3 = [m['pm3_025'] for m in female_sweep_matrices]

            if plottype == 'parameter sweep number':

                upper_f1 = [m['f1_975'] for m in female_sweep_matrices]
                upper_f2 = [m['f2_975'] for m in female_sweep_matrices]
                upper_f3 = [m['f3_975'] for m in female_sweep_matrices]
                upper_m1 = [m['m1_975'] for m in female_sweep_matrices]
                upper_m2 = [m['m2_975'] for m in female_sweep_matrices]
                upper_m3 = [m['m3_975'] for m in female_sweep_matrices]

                lower_f1 = [m['f1_025'] for m in female_sweep_matrices]
                lower_f2 = [m['f2_025'] for m in female_sweep_matrices]
                lower_f3 = [m['f3_025'] for m in female_sweep_matrices]
                lower_m1 = [m['m1_025'] for m in female_sweep_matrices]
                lower_m2 = [m['m2_025'] for m in female_sweep_matrices]
                lower_m3 = [m['m3_025'] for m in female_sweep_matrices]

        # setup standard bounds

        if intervals == 'standard':

            if plottype == 'probability proportion':

                upper_f1 = yval_f1
                upper_f2 = yval_f2
                upper_f3 = yval_f3
                upper_m1 = yval_m1
                upper_m2 = yval_m2
                upper_m3 = yval_m3
                lower_f1 = yval_f1
                lower_f2 = yval_f2
                lower_f3 = yval_f3
                lower_m1 = yval_m1
                lower_m2 = yval_m2
                lower_m3 = yval_m3

            if plottype == 'gender proportion':

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_pct_matrices]))
                upper_f1 = np.where(np.array(upper_f1)>1,1,np.array(upper_f1))
                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_pct_matrices]))
                upper_f2 = np.where(np.array(upper_f2)>1,1,np.array(upper_f2))
                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_pct_matrices]))
                upper_f3 = np.where(np.array(upper_f3)>1,1,np.array(upper_f3))
                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'spct_m1']for m in female_pct_matrices]))
                upper_m1 = np.where(np.array(upper_m1)>1,1,np.array(upper_m1))
                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'spct_m2']for m in female_pct_matrices]))
                upper_m2 = np.where(np.array(upper_m2)>1,1,np.array(upper_m2))
                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'spct_m3']for m in female_pct_matrices]))
                upper_m3 = np.where(np.array(upper_m3)>1,1,np.array(upper_m3))

                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_pct_matrices]))
                lower_f1 = np.where(np.array(lower_f1)<0,0,np.array(lower_f1))
                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_pct_matrices]))
                lower_f2 = np.where(np.array(lower_f2)<0,0,np.array(lower_f2))
                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_pct_matrices]))
                lower_f3 = np.where(np.array(lower_f3)<0,0,np.array(lower_f3))
                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in female_pct_matrices]))
                lower_m1 = np.where(np.array(lower_m1)<0,0,np.array(lower_m1))
                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in female_pct_matrices]))
                lower_m2 = np.where(np.array(lower_m2)<0,0,np.array(lower_m2))
                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in female_pct_matrices]))
                lower_m3 = np.where(np.array(lower_m3)<0,0,np.array(lower_m3))

            if plottype == 'gender number':

                std_matrices = [m.results_matrix[:, 'std_f1':'std_m3'] for m in
                                self.mlist]
                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in std_matrices]))
                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in std_matrices]))
                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in std_matrices]))
                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in std_matrices]))
                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in std_matrices]))
                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in std_matrices]))

                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in std_matrices]))
                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in std_matrices]))
                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in std_matrices]))
                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in std_matrices]))
                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in std_matrices]))
                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in std_matrices]))

            if plottype == 'parameter sweep percentage ':

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_sweep_matrices]))
                upper_f1 = np.where(np.array(upper_f1)>1,1,np.array(upper_f1))
                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_sweep_matrices]))
                upper_f2 = np.where(np.array(upper_f2)>1,1,np.array(upper_f2))
                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_sweep_matrices]))
                upper_f3 = np.where(np.array(upper_f3)>1,1,np.array(upper_f3))
                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'spct_m1']for m in female_sweep_matrices]))
                upper_m1 = np.where(np.array(upper_m1)>1,1,np.array(upper_m1))
                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'spct_m2']for m in female_sweep_matrices]))
                upper_m2 = np.where(np.array(upper_m2)>1,1,np.array(upper_m2))
                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'spct_m3']for m in female_sweep_matrices]))
                upper_m3 = np.where(np.array(upper_m3)>1,1,np.array(upper_m3))


                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'spct_f1'] for m in female_sweep_matrices]))
                lower_f1 = np.where(np.array(lower_f1)<0,0,np.array(lower_f1))
                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'spct_f2'] for m in female_sweep_matrices]))
                lower_f2 = np.where(np.array(lower_f2)<0,0,np.array(lower_f2))
                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'spct_f3'] for m in female_sweep_matrices]))
                lower_f3 = np.where(np.array(lower_f3)<0,0,np.array(lower_f3))
                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'spct_m1'] for m in female_sweep_matrices]))
                lower_m1 = np.where(np.array(lower_m1)<0,0,np.array(lower_m1))
                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'spct_m2'] for m in female_sweep_matrices]))
                lower_m2 = np.where(np.array(lower_m2)<0,0,np.array(lower_m2))
                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'spct_m3'] for m in female_sweep_matrices]))
                lower_m3 = np.where(np.array(lower_m3)<0,0,np.array(lower_m3))


            if plottype == 'parameter sweep number':

                upper_f1 = list(map(add, [y for y in yval_f1], [1.96 * m[
                    'std_f1'] for m in female_sweep_matrices]))
                upper_f2 = list(map(add, [y for y in yval_f2], [1.96 * m[
                    'std_f2'] for m in female_sweep_matrices]))
                upper_f3 = list(map(add, [y for y in yval_f3], [1.96 * m[
                    'std_f3'] for m in female_sweep_matrices]))
                upper_m1 = list(map(add, [y for y in yval_m1], [1.96 * m[
                    'std_m1']for m in female_sweep_matrices]))
                upper_m2 = list(map(add, [y for y in yval_m2], [1.96 * m[
                    'std_m2']for m in female_sweep_matrices]))
                upper_m3 = list(map(add, [y for y in yval_m3], [1.96 * m[
                    'std_m3']for m in female_sweep_matrices]))

                lower_f1 = list(map(sub, [y for y in yval_f1], [1.96 * m[
                    'std_f1'] for m in female_sweep_matrices]))
                lower_f1 = np.where(np.array(lower_f1)<0,0,np.array(lower_f1))
                lower_f2 = list(map(sub, [y for y in yval_f2], [1.96 * m[
                    'std_f2'] for m in female_sweep_matrices]))
                lower_f2 = np.where(np.array(lower_f2)<0,0,np.array(lower_f2))
                lower_f3 = list(map(sub, [y for y in yval_f3], [1.96 * m[
                    'std_f3'] for m in female_sweep_matrices]))
                lower_f3 = np.where(np.array(lower_f3)<0,0,np.array(lower_f3))
                lower_m1 = list(map(sub, [y for y in yval_m1], [1.96 * m[
                    'std_m1'] for m in female_sweep_matrices]))
                lower_m1 = np.where(np.array(lower_m1)<0,0,np.array(lower_m1))
                lower_m2 = list(map(sub, [y for y in yval_m2], [1.96 * m[
                    'std_m2'] for m in female_sweep_matrices]))
                lower_m2 = np.where(np.array(lower_m2)<0,0,np.array(lower_m2))
                lower_m3 = list(map(sub, [y for y in yval_m3], [1.96 * m[
                    'std_m3'] for m in female_sweep_matrices]))
                lower_m3 = np.where(np.array(lower_m3)<0,0,np.array(lower_m3))

        levels = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']
        yvals = [yval_f1, yval_f2, yval_f3, yval_m1, yval_m2, yval_m3]
        dvals = [dval_f1, dval_f2, dval_f3, dval_m1, dval_m2, dval_m3]
        upper_fill = [upper_f1, upper_f2, upper_f3, upper_m1, upper_m2,
                      upper_m3]
        lower_fill = [lower_f1, lower_f2, lower_f3, lower_m1, lower_m2,
                      lower_m3]

        plots = []

        x_data = np.asarray(xval)
        band_x = np.append(x_data, x_data[::-1])

        for key, val in enumerate(levels):

            plots.append(figure(title=titles[key],
                                x_axis_label=xlabels[key],
                                y_axis_label = ylabels[key],
                                width=width_,
                                height=height_))

        for k,v in enumerate(self.mlist):

            for i, p in enumerate(plots):
                p.line(xval, yvals[i][k],
                       line_width=line_width,
                       line_color=linecolor[k],
                       legend=model_legend_label[k])

                band_y = np.append(lower_fill[i][k], upper_fill[i][k][::-1])

                p.patch(band_x,
                        band_y,
                        color=linecolor[k],
                        fill_alpha=transparency)

        for k, v in enumerate(self.mlist):

            if plottype in ['parameter sweep percentage',
                            'parameter sweep probability',
                            'parameter sweep number']:
                for i, p in enumerate(plots):

                    p.line(vertLineForSweepPlot, np.linspace(0, max(band_y)),
                           line_width = vertical_line_width,
                           line_color = vertical_line_color[k],
                           legend = vertical_line_label,
                           line_dash=[6,6])

        if target_plot and plottype != 'gender number':

            for i, p in enumerate(plots):

                p.line(xval,
                       target,
                       line_color=target_color,
                       line_width=target_plot_linewidth,
                       legend=target_plot_legend_label,
                       line_dash=[6,6])

        if target_plot and plottype == 'gender number':
            for k, v in enumerate(self.mlist):
                for i, p in enumerate(plots):
                    p.line(xval,
                           num_targets[i][k],
                           line_color=target_number_color[k],
                           line_width=target_plot_linewidth,
                           legend=target_number_labels[k],
                           line_dash=[6, 6]
                           )

        if percent_line_plot:

            for i, p in enumerate(plots):
                p.line(xval, percent_line_value,
                       line_color=color_percent_line,
                       line_width=percent_linewidth,
                       line_dash=[2,2],
                       legend=percent_legend_label)


        if data_plot:

            for i, p in enumerate(plots):
                p.line(xval, dvals[i],
                       line_color=data_line_color[0],
                       line_width=line_width,
                       line_dash='dashdot',
                       legend=data_line_legend_label[0])


        grid = gridplot([[plots[0], plots[1], plots[2]],
                         [plots[3], plots[4], plots[5]]])




    def export_model_run(self, model_label, model_choice, number_of_runs):

        if not hasattr(self, 'res'):
            self.mlist[0].run_multiple(number_of_runs)

        # first I will allocate the memory by creating an empty dataframe.
        # then I will iterate over the res_array matrix and write to the
        # correct rows of the dataframe. This is more memory efficient compared
        # to appending to a dataframe.

        columnnames = ['run', 'year'] + MODEL_RUN_COLUMNS + \
                      EXPORT_COLUMNS_FOR_CSV + ['model_name', 'date_generated']

        print_array = np.zeros([self.mlist[0].duration * number_of_runs,
                                len(columnnames)])

        for idx in range(number_of_runs):
            print_array[(idx * self.mlist[0].duration):(idx * self.mlist[0].
                duration + self.mlist[0].duration), 0] = idx

            print_array[(idx * self.mlist[0].duration):(idx * self.mlist[0].
                duration + self.mlist[0].duration),
                1:-2] = pd.DataFrame(self.mlist[0].res_array['run'][idx])


        filename = model_label + "_iter" + str(number_of_runs) + ".csv"

        df_print_array = pd.DataFrame(print_array, columns=columnnames).round(2)
        df_print_array.iloc[:, -2] = model_choice
        df_print_array.iloc[:,-1] = str(datetime.datetime.now())
        df_print_array.to_csv(filename, index=False)

        filename = model_label + "_iter" + str(number_of_runs) + "_all_summary.csv"
        results = pd.concat([self.mlist[0].results_matrix.iloc[:, 0:-1].round(2),
                   self.mlist[0].pct_female_matrix.iloc[:, 1:].astype(float).round(3)], axis=1)
        results['date_generated'] = str(datetime.datetime.now())
        results['model_name'] = model_label
        results['year'] = np.arange(self.mlist[0].duration)
        results.to_csv(filename, index=False)

    def plot_attrition_overall(self, settings):
        self.run_all_models(settings['number_of_runs'])
        return PlotComposerOverallAttrition(self.mlist, settings).execute_plot()

    def plot_hiring_mf_overall(self, settings):
        self.run_all_models(settings['number_of_runs'])
        return PlotComposerOverallMFHiring(self.mlist, settings).execute_plot()

    def plot_autocor_dept_size(self, settings):
        #self.run_all_models(settings['number_of_runs'])
        return PlotComposerAutoCorDeptSize(self.mlist, settings).execute_plot()

    def plot_hiring_bylevel(self, settings):
        self.run_all_models(settings['number_of_runs'])
        return PlotComposerLevelHiring(self.mlist, settings).execute_plot()






