%{ 
Authors:    Armin Norouzi(arminnorouzi2016@gmail.com),            
            David Gordon(dgordon@ualberta.ca),
            Eugen Nuss(e.nuss@irt.rwth-aachen.de)
            Alexander Winkler(winkler_a@mmp.rwth-aachen.de)
            Vasu Shamra(vasu3@ualberta.ca),


Copyright 2024 MECE,University of Alberta,
               Teaching and Research 
               Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
%}

clear
close all
clc

%% Set PWD & Path
% has to be executed in scripts folder, where this script here lies
addpath('../Functions/') %for mygrustateFnc etc.

%% Settings
do_training = true;
overwrite_par = true;
break_loop = true;
plot_vars = false; % outputs, inputs, all histograms on data
plot_vars_datasets = false;
plot_explainability = false;
plot_pred_val = false;
plot_pred_test = true;
plot_init = false; % plotting to investigate dataset
plot_init_measurements = false; % look on lvl 1 and lvl2 counters of the individual measurements
kill_violated_data = false; % kill lvl1 and lvl2 hits datapoints
kill_points_zero_h2_doi = true; % when safety lvl 2 was hit, standard controlelr was on until the end of the feq cycle, so kill these cycles here
save_plots_sw = false;
save_analysis = true;
verify_my_func = true;
plot_step_zoom = false;
plot_training_loss = false;

% plotting options
Opts.multi_lines_ylabel = true;
Opts.LatexLabels = true;
Opts.fontsize = 11; % diss needs 13, should have no influence though
Opts.ltx_tma = '\tMa';
Opts.ltx_ptom = '\tPtoM';
Opts.ltx_ama = '\aMa';
Opts.ltx_thy = '\tHy';
Opts.ltx_pme = '\pMe';
Opts.ltx_cnox = '\cNox';
Opts.ltx_cpm = '\cPm';
Opts.ltx_dpm = '\dpm';
fntsze = Opts.fontsize;
tick_step = 500;

% training
ratio_train = 0.8;
ratio_val = 0.95;

MP = 2024;
trainingrun = 271; no_fb = false; % fb, final model integrated into mpc, see results folder
% trainingrun = 282; no_fb = true ; % no fb final model integrated into mpc, see results folder

%% Load data
load('Test_480_to_481_conc.mat')
data_conc.fpga_lastCycle_MPRR = 10*data_conc.fpga_lastCycle_MPRR; % dp max, in Mpa per CAD
[utrain_4801, ytrain_4801, uval_4801, yval_4801, utest_4801, ytest_4801, utotal_4801, ytotal_4801] = getDatasets_480(data_conc, 1, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi);

load('Test_502_NoPress.mat')
[utrain_502, ytrain_502, uval_502, yval_502, utest_502, ytest_502, utotal_502, ytotal_502] = getDatasets_480(Test_502_NoPress, 1, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi);

load('Test_507_NoPress.mat')
[utrain_507, ytrain_507, uval_507, yval_507, utest_507, ytest_507, utotal_507, ytotal_507] = getDatasets_480(Test_507_NoPress, 1, ratio_train, ratio_val, plot_init_measurements, kill_violated_data, kill_points_zero_h2_doi);

savename_data = '2024_480_to_507_GRU_normalized_post.mat';
savename_datasets = '2024_480_to_507_GRU_normalized_datasets.mat';
savename_datasets_phys = '2024_480_to_507_GRU_phys_datasets.mat';

%concatenate
utrain = [utrain_4801'; utrain_502'; utrain_507'];
ytrain = [ytrain_4801'; ytrain_502'; ytrain_507'];

uval = [uval_4801'; uval_502'; uval_507'];
yval = [yval_4801'; yval_502'; yval_507'];

utest = [utest_4801'; utest_502'; utest_507'];
ytest = [ytest_4801'; ytest_502'; ytest_507'];

utotal = [utotal_4801'; utotal_502'; utotal_507'];
ytotal = [ytotal_4801'; ytotal_502'; ytotal_507'];
% utotal_sorted = [utrain; uval; utest];
% ytotal_sorted = [ytrain; yval; ytest];

% get dat from concatenated datasets
DOI_main_cycle = utotal(:,1)';
P2M_cycle = utotal(:,2)';
SOI_main_cycle = utotal(:,3)';
H2_doi_cycle = utotal(:,4)'; % convert s to ms
IMEP_old = utotal(:,5)';
IMEP_cycle = ytotal(:,1)'; % pressure, bascically load in MPa
NOx_cycle = ytotal(:,2)'; % cheap CAN sensor, not FTIR! in ppm
Soot_cycle = ytotal(:,3)'; % in mgm3
MPRR_cycle = ytotal(:,4)'; % dp max, in Mpa per CAD

%% mkdir
mkdir('../Plots/'+ sprintf("%04d",MP)',sprintf('%04d',trainingrun))
mkdir('../','/Results/');

max_scale = length(utotal); % 65024 for VSR1123002

%% if plot_vars (plot inputs, outputs, histogram whole dataset)
if plot_vars

%% ploting inputs whole dataset
figure
set(gcf, 'Position', [100, 100, 1800, 1000]);
set(gcf,'color','w');

ax5=subplot(4,1,4);
plot(H2_doi_cycle*1e3, 'k','LineWidth',1.0)
grid on
% ylabel({'H2 DOI', '/ ms'},'Interpreter','latex')
xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'H2 DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        ylabel('H2 DOI / ms','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')


ax6=subplot(4,1,1);
plot(DOI_main_cycle*1e3, 'k','LineWidth',2)
grid on
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'Main Inj. DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        ylabel('Main Inj. DOI / ms','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];

ax7=subplot(4,1,2);
plot(P2M_cycle, 'k','LineWidth',1.0)
grid on
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ \\ / \mu s')},'Interpreter','latex')
    else
        ylabel({'P2M';'/ us'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        ylabel('P2M / us','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];

ax8=subplot(4,1,3);
plot(SOI_main_cycle, 'k','LineWidth',1.0)
grid on
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ \\ / CADbTDC')},'Interpreter','latex')
    else
        ylabel({'SOI Main';'/ CADbTDC'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        ylabel('SOI Main / CADbTDC','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];

set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Inputs"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end


%% plotting outputs whole dataset
figure
set(gcf, 'Position', [100, 100, 1800, 800]);
set(gcf,'color','w');

%--------------------------------------------------
ax1=subplot(4,1,1);
plot(IMEP_cycle * 1e-5, 'k','LineWidth',1.0)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'IMEP',' / bar'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP';' / bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP / bar','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];
%--------------------------------------------------
ax2=subplot(4,1,2);
plot(NOx_cycle, 'k','LineWidth',1.0)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'NO$_x$', '/ ppm'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ \\ / ppm')},'Interpreter','latex')
    else
        ylabel({'NO$_x$';'/ ppm'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        ylabel('NO$_x$ / ppm','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];
%--------------------------------------------------
ax3=subplot(4,1,3);
plot(Soot_cycle, 'k','LineWidth',1.0)
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'Soot',' / (mg/m$^3$)'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ \\ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel({'Soot';'/ (mg/m$^3$)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];
%--------------------------------------------------
ax4=subplot(4,1,4);
plot(MPRR_cycle  * 1e-5, 'k','LineWidth',0.5)
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'MPRR',' / (bar/CAD)'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ \\ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel({'MPRR';'/ (bar/CAD)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel('MPRR / (bar/CAD)','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
% ax.XTickLabel = [];

linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8],'x');

set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/Outputs"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end

%% common plot for inputs and outputs
figure
set(gcf, 'Position', [100, 100, 1800, 1000]);
set(gcf,'color','w');

ax6=subplot(8,1,1);
plot(DOI_main_cycle*1e3, 'k','LineWidth', 1.0, 'DisplayName', 'Measurement');
hold on;
xline(69995, '--g', 'LineWidth', 1.5, 'DisplayName', 'Hydrogen Cylinder Change');
xline(94887, '--g', 'LineWidth', 1.5, 'HandleVisibility', 'off');  % Same label, so don't duplicate in legend
legend('Interpreter', 'latex', 'Location','southeast','Orientation','horizontal')
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'Main Inj. DOI', '/ ms'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'Main Inj. DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_tma,'$ / ms')},'Interpreter','latex')
    else
        ylabel('Main Inj. DOI / ms','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];

ax7=subplot(8,1,2);
plot(P2M_cycle, 'k','LineWidth',0.5);
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'P2M', '/ us'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ \\ / \mu s')},'Interpreter','latex')
    else
        ylabel({'P2M';'/ us'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ptom,'$ / \mu s')},'Interpreter','latex')
    else
        ylabel('P2M / us','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];

ax8=subplot(8,1,3);
plot(SOI_main_cycle, 'k','LineWidth',0.5); 
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'SOI Main','/ (bTDC CAD)'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ \\ / CADbTDC')},'Interpreter','latex')
    else
        ylabel({'SOI Main';'/ CADbTDC'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_ama,'$ / CADbTDC')},'Interpreter','latex')
    else
        ylabel('SOI Main / CADbTDC','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];

set(gcf,'units','points','position',[200,200,900,400])

ax5=subplot(8,1,4);
plot(H2_doi_cycle*1e3, 'k','LineWidth',0.5); 
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
% ylabel({'H2 DOI', '/ ms'},'Interpreter','latex')
% xlabel("Cycles / -",'Interpreter', 'latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ \\ / ms')},'Interpreter','latex')
    else
        ylabel({'H2 DOI';'/ ms'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_thy,'$ / ms')},'Interpreter','latex')
    else
        ylabel('H2 DOI / ms','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
% set(gca,'XTick', 0:tick_step:5000)
ax.XTickLabel = [];
ax5 = gca;

%--------------------------------------------------
ax1=subplot(8,1,5);
plot(IMEP_cycle * 1e-5, 'k','LineWidth',0.5);
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'IMEP',' / bar'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP / bar','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];
%--------------------------------------------------
ax2=subplot(8,1,6);
plot(NOx_cycle, 'k','LineWidth',0.5); 
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'NO$_x$', '/ ppm'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ \\ / ppm')},'Interpreter','latex')
    else
        ylabel({'NO$_x$';'/ ppm'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        ylabel('NO$_x$ / ppm','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];
%--------------------------------------------------
ax3=subplot(8,1,7);
plot(Soot_cycle, 'k','LineWidth',0.5);
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
% xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'Soot',' / (mg/m$^3$)'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ \\ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel({'Soot';'/ (mg/m$^3$)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end
end
xlim([0,max_scale])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
ax.XTickLabel = [];
%--------------------------------------------------
ax4=subplot(8,1,8);
plot(MPRR_cycle  * 1e-5, 'k','LineWidth',0.5);
hold on;
xline(69995, '--g', 'LineWidth', 1.5);
xline(94887, '--g', 'LineWidth', 1.5);
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
% ylabel({'MPRR',' / (bar/CAD)'},'Interpreter','latex')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ \\ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel({'MPRR';'/ (bar/CAD)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel('MPRR / (bar/CAD)','Interpreter','latex')
    end
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
set(gca,'fontsize', fntsze)
set(gca,'TickLabelInterpreter','latex')
% ax.XTickLabel = [];

% linkaxes([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8],'x');

set(gcf,'units','points','position',[200,200,900,400])

if save_plots_sw
    type = "/InputsOutputs"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 5, Opts.multi_lines_ylabel)
end

%% Histogram on whole dataset
figure
set(gcf,'color','w');
histogram(IMEP_cycle * 1e-5)
grid on

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        xlabel('IMEP / bar','Interpreter','latex')
    end

ylabel("Count / -",'Interpreter', 'latex')
% title('IMEP Data Distribution','Interpreter', 'latex') %%%%
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

if save_plots_sw
    type = "/IMEP_Data_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

figure
set(gcf,'color','w');
histogram(NOx_cycle) 
grid on

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        xlabel('NO$_x$ / ppm','Interpreter','latex')
    end

ylabel("Count / -",'Interpreter', 'latex')
% title('NOX Data Distribution','Interpreter', 'latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

if save_plots_sw
    type = "/NOX_Data_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

figure
set(gcf,'color','w');
histogram(Soot_cycle)
grid on
ylabel({'Count / -'},'Interpreter','latex')

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        xlabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end

% title('Soot Data Distribution','Interpreter', 'latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

if save_plots_sw
    type = "/SOOT_Data_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

figure
set(gcf,'color','w');
histogram(MPRR_cycle * 1e-5)
grid on

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        xlabel('MPRR / (bar/CAD)','Interpreter','latex')
    end

ylabel("Count / -",'Interpreter', 'latex')
% title('MPRR Data Distribution','Interpreter', 'latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

if save_plots_sw
    type = "/MPRR_Data_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

%% Histogram on whole dataset
figure
subplot(2,2,1)
set(gcf,'color','w');
histogram(IMEP_cycle * 1e-5)
grid on
    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        xlabel('IMEP / bar','Interpreter','latex')
    end
ylabel("Count / -",'Interpreter', 'latex')
% title('IMEP Data Distribution','Interpreter', 'latex') %%%%
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,2)
set(gcf,'color','w');
histogram(NOx_cycle) 
grid on

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        xlabel('NO$_x$ / ppm','Interpreter','latex')
    end

ylabel("Count / -",'Interpreter', 'latex')
% title('NOX Data Distribution','Interpreter', 'latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,3)
set(gcf,'color','w');
histogram(Soot_cycle)
grid on
ylabel({'Count / -'},'Interpreter','latex')
    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        xlabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end
% title('Soot Data Distribution','Interpreter', 'latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

subplot(2,2,4)
set(gcf,'color','w');
histogram(MPRR_cycle * 1e-5)
grid on
    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        xlabel('MPRR / (bar/CAD)','Interpreter','latex')
    end
ylabel("Count / -",'Interpreter', 'latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')

if save_plots_sw
    type = "/Outputs_Data_Distribution"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

end

%% analysis array init
runs_total_max = 30;
analysis = struct();
analysis.FinalRMSE = zeros(runs_total_max, 1);
analysis.FinalValidationLoss = zeros(runs_total_max, 1);
analysis.TotalLearnables = zeros(runs_total_max, 1);
analysis.ElapsedTime = zeros(runs_total_max, 1);
A = ['XXXX_00YY_00YY_0ZZZ.mat'];
analysis.savename = repmat(A, runs_total_max, 1);
run_nmbr = 0;


%% plot histograms and steps on the different datasets (train, val, test)
if plot_vars_datasets == true 

IMEP_cycle_tr = ytrain(:,1);
IMEP_cycle_val = yval(:,1);
IMEP_cycle_test = ytest(:,1);
NOx_cycle_tr = ytrain(:,2);
NOx_cycle_val = yval(:,2);
NOx_cycle_test = ytest(:,2);
Soot_cycle_tr = ytrain(:,3);
Soot_cycle_val = yval(:,3);
Soot_cycle_test = ytest(:,3);
MPRR_cycle_tr = ytrain(:,4);
MPRR_cycle_val = yval(:,4);
MPRR_cycle_test = ytest(:,4);
max_scale_imep = max( [max(ytrain(:,1)), max(yval(:,1)), max(ytest(:,1))])  * 1e-5;
max_scale_nox = max( [max(ytrain(:,2)), max(yval(:,2)), max(ytest(:,2))]);
max_scale_soot = max( [max(ytrain(:,3)), max(yval(:,3)), max(ytest(:,3))]);
max_scale_mprr = max( [max(ytrain(:,4)), max(yval(:,4)), max(ytest(:,4))]) * 1e-5;

% start plotting
% imep plots
fig = figure;
subplot(3,1,1)
set(gcf,'color','w');
histogram(IMEP_cycle_tr * 1e-5)
grid on
xlim([0,max_scale_imep])
title('Training Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,2)
set(gcf,'color','w');
histogram(IMEP_cycle_val * 1e-5)
grid on
xlim([0,max_scale_imep])
ylabel("Count / -",'Interpreter', 'latex')
title('Validation Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,3)
set(gcf,'color','w');
histogram(IMEP_cycle_test * 1e-5)
grid on
xlim([0,max_scale_imep])

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        xlabel('IMEP / bar','Interpreter','latex')
    end
title('Test Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;   % ax.XTickLabel = [];
% linkaxes([ax ax1 ax2],'x')


if save_plots_sw
    type = "/IMEP_Data_Distribution_sets"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

% NOX plots
fig = figure;
subplot(3,1,1)
set(gcf,'color','w');
histogram(NOx_cycle_tr)
grid on
xlim([0,max_scale_nox])
title('Training Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,2)
set(gcf,'color','w');
histogram(NOx_cycle_val)
grid on
xlim([0,max_scale_nox])
ylabel("Count / -",'Interpreter', 'latex')
title('Validation Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,3)
set(gcf,'color','w');
histogram(NOx_cycle_test )
grid on
xlim([0,max_scale_nox])
    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        xlabel('NO$_x$ / ppm','Interpreter','latex')
    end
title('Test Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; % ax.XTickLabel = [];
% linkaxes([ax ax1 ax2],'x')

if save_plots_sw
    type = "/NOx_Data_Distribution_sets"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

% Soot plots
fig = figure;
subplot(3,1,1)
set(gcf,'color','w');
histogram(Soot_cycle_tr)
grid on
xlim([0,max_scale_soot])
title('Training Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,2)
set(gcf,'color','w');
histogram(Soot_cycle_val)
grid on
xlim([0,max_scale_soot])
ylabel("Count / -",'Interpreter', 'latex')
title('Validation Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,3)
set(gcf,'color','w');
histogram(Soot_cycle_test )
grid on
xlim([0,max_scale_soot])

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        xlabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end

title('Test Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; % ax.XTickLabel = [];
% linkaxes([ax ax1 ax2],'x')

if save_plots_sw
    type = "/Soot_Data_Distribution_sets"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

% MPRR plots
fig = figure;
subplot(3,1,1)
set(gcf,'color','w');
histogram(MPRR_cycle_tr *1e-5)
grid on
xlim([0,max_scale_mprr])
title('Training Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,2)
set(gcf,'color','w');
histogram(MPRR_cycle_val *1e-5)
grid on
xlim([0,max_scale_mprr])
ylabel("Count / -",'Interpreter', 'latex')
title('Validation Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; ax.XTickLabel = [];

subplot(3,1,3)
set(gcf,'color','w');
histogram(MPRR_cycle_test *1e-5)
grid on
xlim([0,max_scale_mprr])

    if Opts.LatexLabels
        xlabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        xlabel('MPRR / (bar/CAD)','Interpreter','latex')
    end

title('Test Dataset','Interpreter', 'latex')
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0; % ax.XTickLabel = [];
% linkaxes([ax ax1 ax2],'x')

if save_plots_sw
    type = "/MPRR_Data_Distribution_sets"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
end

end
%% Normalizing data - ONLY with complete dataset!
[~, u1_min, u1_range] = dataTrainNormalize(DOI_main_cycle');
[~, u2_min, u2_range] = dataTrainNormalize(P2M_cycle');
[~, u3_min, u3_range] = dataTrainNormalize(SOI_main_cycle');
[~, u4_min, u4_range] = dataTrainNormalize(H2_doi_cycle');
% [~, u5_min, u5_range] = dataTrainNormalize(IMEP_old'); % feedback of old IMEP

[~, y1_min, y1_range] = dataTrainNormalize(IMEP_cycle');
[~, y2_min, y2_range] = dataTrainNormalize(NOx_cycle');
[~, y3_min, y3_range] = dataTrainNormalize(Soot_cycle');
[~, y4_min, y4_range] = dataTrainNormalize(MPRR_cycle');

utrain_1 = normalize_var(utrain(:,1), ...
    u1_min, u1_range, 'to-scaled');
utrain_2 = normalize_var(utrain(:,2), ...
    u2_min, u2_range, 'to-scaled');
utrain_3 = normalize_var(utrain(:,3), ...
    u3_min, u3_range, 'to-scaled');
utrain_4 = normalize_var(utrain(:,4), ...
    u4_min, u4_range, 'to-scaled');
utrain_5 = normalize_var(utrain(:,5), ...
    y1_min, y1_range, 'to-scaled');
ytrain_1 = normalize_var(ytrain(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytrain_2 = normalize_var(ytrain(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytrain_3 = normalize_var(ytrain(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytrain_4 = normalize_var(ytrain(:,4), ...
    y4_min, y4_range, 'to-scaled');

uval_1 = normalize_var(uval(:,1), ...
    u1_min, u1_range, 'to-scaled');
uval_2 = normalize_var(uval(:,2), ...
    u2_min, u2_range, 'to-scaled');
uval_3 = normalize_var(uval(:,3), ...
    u3_min, u3_range, 'to-scaled');
uval_4 = normalize_var(uval(:,4), ...
    u4_min, u4_range, 'to-scaled');
uval_5 = normalize_var(uval(:,5), ...
    y1_min, y1_range, 'to-scaled');
yval_1 = normalize_var(yval(:,1), ...
    y1_min, y1_range, 'to-scaled');
yval_2 = normalize_var(yval(:,2), ...
    y2_min, y2_range, 'to-scaled');
yval_3 = normalize_var(yval(:,3), ...
    y3_min, y3_range, 'to-scaled');
yval_4 = normalize_var(yval(:,4), ...
    y4_min, y4_range, 'to-scaled');

utest_1 = normalize_var(utest(:,1), ...
    u1_min, u1_range, 'to-scaled');
utest_2 = normalize_var(utest(:,2), ...
    u2_min, u2_range, 'to-scaled');
utest_3 = normalize_var(utest(:,3), ...
    u3_min, u3_range, 'to-scaled');
utest_4 = normalize_var(utest(:,4), ...
    u4_min, u4_range, 'to-scaled');
utest_5 = normalize_var(utest(:,5), ...
    y1_min, y1_range, 'to-scaled');
ytest_1 = normalize_var(ytest(:,1), ...
    y1_min, y1_range, 'to-scaled');
ytest_2 = normalize_var(ytest(:,2), ...
    y2_min, y2_range, 'to-scaled');
ytest_3 = normalize_var(ytest(:,3), ...
    y3_min, y3_range, 'to-scaled');
ytest_4 = normalize_var(ytest(:,4), ...
    y4_min, y4_range, 'to-scaled');


%% Dateset Definition
if no_fb
    % without IMEP feedback u5
    utrain_load = [utrain_1'; utrain_2'; utrain_3'; utrain_4'];
    ytrain_load = [ytrain_1'; ytrain_2'; ytrain_3'; ytrain_4'];
    
    uval_load = [uval_1'; uval_2'; uval_3'; uval_4'];
    yval_load = [yval_1'; yval_2'; yval_3'; yval_4'];
    
    utest_load = [utest_1'; utest_2'; utest_3'; utest_4'];
    ytest_load = [ytest_1'; ytest_2'; ytest_3'; ytest_4'];
else
    % with IMEP feedback u5
    utrain_load = [utrain_1'; utrain_2'; utrain_3'; utrain_4'; utrain_5'];
    ytrain_load = [ytrain_1'; ytrain_2'; ytrain_3'; ytrain_4'];
    
    uval_load = [uval_1'; uval_2'; uval_3'; uval_4'; uval_5'];
    yval_load = [yval_1'; yval_2'; yval_3'; yval_4'];
    
    utest_load = [utest_1'; utest_2'; utest_3'; utest_4'; utest_5'];
    ytest_load = [ytest_1'; ytest_2'; ytest_3'; ytest_4'];

end


%% Save Datafiles
Data = struct();
Data.label = {'imep'; 'nox'; 'soot'; 'mprr'; ...
    'doi_main'; 'p2m'; 'soi_main'; 'doi_h2'; 'imep_old'};
data = {IMEP_cycle', NOx_cycle', Soot_cycle', MPRR_cycle', ...
    DOI_main_cycle', P2M_cycle', SOI_main_cycle', H2_doi_cycle', IMEP_old'};
for ii = 1:length(data)
    [Data.signal{ii, 1}, Data.mean{ii, 1}, Data.std{ii, 1}] = ...
        normalize_data(data{ii});
end

label = Data.label; mean = Data.mean; std = Data.std; signal = Data.signal;
save(fullfile(['../Results/',savename_data]), 'label', 'mean', 'std', 'signal');
clear mean; clear std; 

DataSets = struct();
DataSets.utrain_load = utrain_load;
DataSets.ytrain_load = ytrain_load;
DataSets.uval_load = uval_load;
DataSets.yval_load = yval_load;
DataSets.utest_load = utest_load;
DataSets.ytest_load = ytest_load;
DataSets.ratio_train = ratio_train;
DataSets.ratio_val = ratio_val;
save(fullfile(['../Results/',savename_datasets]), 'DataSets');

DataSetsPhys = struct();
DataSetsPhys.utrain = utrain;
DataSetsPhys.ytrain = ytrain;
DataSetsPhys.uval = uval;
DataSetsPhys.yval = yval;
DataSetsPhys.utest = utest;
DataSetsPhys.ytest = ytest;
DataSetsPhys.ratio_train = ratio_train;
DataSetsPhys.ratio_val = ratio_val;
DataSetsPhys.label = Data.label; DataSetsPhys.mean = Data.mean; DataSetsPhys.std = Data.std;
save(fullfile(['../Results/',savename_datasets_phys]), 'DataSetsPhys');


%% Training 
for numHiddenUnits1 = [8] % [8,8,16,16] % for loop for grid searhc / to try out different units number within the FF layer IMPORTANT PARAMETER
for LSTMStateNum= [8] % [8,16] % for loop for grid searhc / to try out different units number within the recurrent layer IMPORTANT PARAMETER
tic

run_nmbr = run_nmbr + 1;
disp ( ['Measurement Point / Save File Number ', num2str(trainingrun)] );
disp ( ['Grid Search Number Iteration ', num2str(run_nmbr)] );

% Model inputs:
%     -DOIMain
%     -SOI pre
%     -SOI Main
%     -H2 DOI
%     -IMEP (last) (if feedback active)
% Model Outputs:
%     -IMEP
%     -NOx
%     -Soot
%     -MPRR

% mat = [u1,u2,u3,u4,u5];
% plotmatrix(mat)

%% Create Newtwork arch + setting / options
numResponses = 4; % y1 y2 y3 y4

if no_fb
    featureDimension = 4; % u1 u2 u3 u4 u5
else
    featureDimension = 5; % u1 u2 u3 u4 u5 % with feedback imep
end

maxEpochs = 3500; % IMPORTANT PARAMETER
miniBatchSize = 512; % IMPORTANT PARAMETER

% architecture
Networklayer_h2df = [...
    sequenceInputLayer(featureDimension)
    fullyConnectedLayer(4*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(4*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(8*numHiddenUnits1)
    reluLayer
    gruLayer(LSTMStateNum,'OutputMode','sequence',InputWeightsInitializer='he',RecurrentWeightsInitializer='he')
    fullyConnectedLayer(8*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(4*numHiddenUnits1)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

% training options
options_tr = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'SequenceLength',8192,...
    'Shuffle','once', ...
    'Plots','training-progress',...
    'Verbose',1, ...
    'VerboseFrequency',64,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',250,...
    'LearnRateDropFactor',0.75,...
    'L2Regularization',0.1,...
    'ValidationFrequency',10,...
    'InitialLearnRate', 0.001,...
    'Verbose', false, ...
    'ExecutionEnvironment', 'cpu', ...
    'ValidationData',[{uval_load} {yval_load}],...
    'OutputNetwork','best-validation-loss');

%% training and Saving model data
savename = [sprintf('%04d',MP),'_',sprintf('%04d',numHiddenUnits1),'_',sprintf('%04d',LSTMStateNum),'_',sprintf('%04d',trainingrun),'.mat'];

if do_training == true
    tic
    [h2df_model, h2df_model_infor] = trainNetwork(utrain_load,ytrain_load,Networklayer_h2df,options_tr);
    toc
    ElapsedTime = toc;
    h2df_model_analysis = analyzeNetwork(h2df_model); % analysis including total number of learnable parameters
    h2df_model_infor.ElapsedTime = ElapsedTime;

    save(['../Results/h2df_model_',savename],"h2df_model")
    save(['../Results/h2df_model_info_',savename],"h2df_model_infor")
    save(['../Results/h2df_model_analysis_',savename],"h2df_model_analysis")
else
    load(['../Results/h2df_model_',savename])
    load(['../Results/h2df_model_info_',savename])
    load(['../Results/h2df_model_analysis_',savename])
end

%% performance meta data for grid search etc
analysis.FinalRMSE(run_nmbr,1) = h2df_model_infor.FinalValidationRMSE;
analysis.FinalValidationLoss(run_nmbr,1)  = h2df_model_infor.FinalValidationLoss;
analysis.TotalLearnables(run_nmbr,1) = h2df_model_analysis.TotalLearnables;
analysis.ElapsedTime(run_nmbr,1) = h2df_model_infor.ElapsedTime;
analysis.savename(run_nmbr,1:length(savename)) = savename;
% savename

%% Plot Training Results

if plot_training_loss
num_epoch = round(length(h2df_model_infor.TrainingLoss) / 10);
val_freq = 10;
TrainLoss = zeros(1, num_epoch);
% clear mean;
for i = 1 : num_epoch
    TrainLoss(1, i) = mean(h2df_model_infor.TrainingLoss(1+((i-1)*val_freq) : (val_freq-1)+((i-1)*val_freq)));
end
% TrainLoss(1,2:num_epoch) = h2df_model_infor.TrainingLoss(10:10:(num_epoch*10)-10);
ValLoss = zeros(1,num_epoch);
ValLoss(1,1) = h2df_model_infor.ValidationLoss(1);
ValLoss(1,2:num_epoch) = h2df_model_infor.ValidationLoss(1,val_freq:val_freq:(num_epoch*val_freq)-val_freq);

figure
set(gcf,'color','w');
set(gcf,'units','points','position',[200,200,900,400])
% plot(h2df_model_infor.TrainingLoss,"--", 'Color', 'blue','LineWidth',1);
plot(TrainLoss,"--", 'Color', 'blue','LineWidth',1);
hold on
% plot(fillmissing(h2df_model_infor.ValidationLoss,'linear'), 'k','LineWidth',2);
plot(ValLoss, 'k','LineWidth',2);
% best validation loss, mark the epoch with the minimum validation loss and mark it with a cross
plot((h2df_model_infor.OutputNetworkIteration / val_freq), h2df_model_infor.FinalValidationLoss, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
grid on
xlabel("Epochs / -",'Interpreter', 'latex');
ylabel("Loss / -",'Interpreter', 'latex');
legend("Training","Validation","Best Validation Loss - Final Network",'Location','northeast','Orientation','horizontal','Interpreter', 'latex');
set(gcf,'units','points','position',[200,200,900,400])
set(gca,'FontSize',fntsze)
set(gca, 'YScale', 'log')
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;

if save_plots_sw
    type = "/Loss_Iteration";
    resolution = 0;
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, resolution, Opts.multi_lines_ylabel)
end


end

%% Prediction on val dataset
if no_fb
    y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4']) ; % with IMEP
else
    y_hat = predict(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4'; uval_5']) ; % with IMEP
end
y1_hat = y_hat(1,:);
y2_hat = y_hat(2,:);
y3_hat = y_hat(3,:);
y4_hat = y_hat(4,:);

% Denormalize Predictions
IMEP_cycle_hat = dataTraindeNormalize(y1_hat,y1_min,y1_range);
NOx_cycle_hat = dataTraindeNormalize(y2_hat,y2_min,y2_range);
SOOT_cycle_hat = dataTraindeNormalize(y3_hat,y3_min,y3_range);
MPRR_cycle_hat = dataTraindeNormalize(y4_hat,y4_min,y4_range);

%% Explainability on val dataset
if plot_explainability    
    if no_fb
        act = activations(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4'],"gru");
    else
        act = activations(h2df_model,[uval_1'; uval_2'; uval_3'; uval_4';uval_5'],"gru");               
    end    
    step_start = 170;
    step_end = 180;
    activation_data = act{1,1}(1:8,step_start:step_end); % first 8 states, cycles 3 to 14

    % figure; hold on; 
    % heatmap(act{1,1}(1:8,1:100));
    % xlabel("Cycle \ -")
    % ylabel("Hidden Unit")
    % title("GRU Activations")
    % set(gca,'FontSize',fntsze)  

    % First figure - Activation Values (in blues)
    figActivations = figure('Position', [100, 100, 1800, 900],'Color', 'w');
    hold on;      
    % Generate shades of blue
    blues = [
    0.0000, 0.4470, 0.7410; 
    % Default MATLAB blue
    0.3010, 0.7450, 0.9330; 
    % Light blue
    0.0000, 0.3176, 0.6196; 
    % Medium blue
    0.0000, 0.2470, 0.5410; 
    % Darker blue
    0.0000, 0.1840, 0.4196; 
    % Even darker blue
    0.0392, 0.1490, 0.3450; 
    % Deep blue
    0.0784, 0.1137, 0.2706; 
    % Very deep blue
    0.1176, 0.0784, 0.1961 
    % Darkest blue
    ];

    % Plot each activation unit
    x_values = step_start:step_end;
    for i = 1:8
        plot(x_values, activation_data(i,:),'LineWidth', 1.0, ...
        'Color', blues(i,:)); % , 'Marker', 'o', 'MarkerSize', 8
    end
    ylim([-0.4 0.4]);

    % Customize plot appearance
    grid on; box on;
    % xlim([1 12]);
    % Set x-labels explicitly to 0:10    
    xticklabels(arrayfun(@num2str, 0:(step_end-step_start), 'UniformOutput', false));

    % Set axis labels with LaTeX interpreter
    xlabel('Cycles / -', 'Interpreter', 'latex', 'FontSize', fntsze);
    ylabel('GRU States / -', 'Interpreter', 'latex', 'FontSize', fntsze);
    
    % Configure axis properties
    ax = gca;
    set(ax, 'FontSize', fntsze, 'TickLabelInterpreter', 'latex', 'FontName', 'Times New Roman');
    ax.XRuler.Exponent = 0;

    if save_plots_sw
        type = "/GRUactivationstestAct"; 
        save_plots(gcf, MP, trainingrun, type, plot_step_zoom)
    end
    
    % Second figure - Input signals (keep original colors)
    figSignals = figure('Position', [100, 100, 1800, 900], 'Color', 'w');
    hold on;
    % Plot with consistent styling
    p1 = plot(x_values, uval_1(step_start:step_end), 'LineWidth', 1.0, 'Color', [0.85, 0.325, 0.098], 'DisplayName',...
        'DOI Main', 'Marker', 'o', 'MarkerSize', 8);
    p2 = plot(x_values, uval_2(step_start:step_end), 'LineWidth', 1.0, 'Color', 'k', ...
    'DisplayName', 'P2M', 'Marker', '*', 'MarkerSize', 8);
    p3 = plot(x_values, uval_3(step_start:step_end), 'LineWidth', 1.0, 'Color', [0.929, 0.694, 0.125], 'DisplayName',...
        'SOI Main', 'Marker', 's', 'MarkerSize', 8);
    p4 = plot(x_values, uval_4(step_start:step_end), 'LineWidth', 1.0, 'Color', [0.494, 0.184, 0.556], 'DisplayName',...
        'DOI H2', 'Marker', 'd', 'MarkerSize', 8);
    p5 = plot(x_values, uval_5(step_start:step_end), 'LineWidth', 1.0, 'Color', [0.466, 0.674, 0.188], 'DisplayName', ...
        'Last IMEP', 'Marker', '^', 'MarkerSize', 8);
    % Set axis limits
    ylim([0 1.0]);
    % xlim([1 12]);  
    xticklabels(arrayfun(@num2str, 0:(step_end-step_start), 'UniformOutput', false));
    % Customize plot appearance
    grid on; box on;
    % Set axis labels with LaTeX interpreter
    xlabel('Cycles / -', 'Interpreter', 'latex', 'FontSize', fntsze);
    ylabel('Normalized Signal / -', 'Interpreter', 'latex', 'FontSize', fntsze);

    % Configure axis properties
    ax = gca;
    set(ax, 'FontSize', fntsze, 'TickLabelInterpreter','latex', ...
        'FontName', 'Times New Roman');
    ax.XRuler.Exponent = 0;
    % Add legend with consistent styling
    lgd = legend([p1, p2, p3, p4, p5], 'Location', 'northeast', ...
        'Orientation','vertical', 'FontSize', fntsze, 'Interpreter', 'latex', 'NumColumns', 1);
    lgd.Position = [0.80 0.60 0.15 0.35];       

    if save_plots_sw
        type = "/GRUactivationstestSignals"; 
        save_plots(gcf, MP, trainingrun, type, plot_step_zoom, [], Opts.multi_lines_ylabel)
    end                 
end

%% plotting & postprocessing data on val dataset

if plot_pred_val
% errperf(T,P,'mae')
% mse (mean squared error)
% rmse (root mean squared error)
% mspe (mean squared percentage error)
% rmspe (root mean squared percentage error)
%%: beware of conversion and units here! s vs ms for doi main and h2
%%doi!!!!
% prompt: 
% fill this latex table below with the correct values from the second input - the data. Check for the test dataset data (_test). 
% But please multiple MAE, RMSE from DOIm and DOIh2 by 1e3. Multiply the MSE value by 1e6. do not change the r2 and the NRMSE value. 
% Use the e-3 notation instead of the x 10^-3 notation.
% round to two decimal places for all values but for R square / R2, round to four there. 
% 
% Create a second table for the validation dataset (_val).
% 
% 
% 
% \begin{table}%[h!]
% 	\centering
% 	\begin{tabular}{|l|c|c|c|c} 
% 		\hline
% 		\textbf{Metric} &  \textbf{$\tMa$} &  \textbf{$\tPtoM$} &  \textbf{$\aMa$} &  \textbf{$\tHy$} \\
% 		\hline \hline
% 		MAE /  & 7.23 e-3 ms &  us & CAD &  ms \\
% 		\hline
% 		MSE /  &  ms$^2$ & us$^2$ & CAD$^2$ & ms$^2$ \\
% 		\hline
% 		RMSE /  & ms & us & CAD & ms \\
% 		\hline
% 		NRMSE / - & 2.\% & 9.\% & \% & \% \\
% 		\hline
% 		R2 / - & 0.9841 & 0.5617 \\
% 		\hline             
% 	\end{tabular}
% 	\caption[Behavior Cloning deep neural network prediction performance metrics on unseen test dataset.]{Behavior Cloning deep neural network prediction performance metrics on unseen test dataset. MAE: Mean Absolute Error, MSE: Mean Squared Error, RMSE: Root MSE, NRMSE: Normalized RMSE.}
% 	\label{tab:bc_train_metrics_test}
% \end{table}
% 
%  % get from excel copy paste and transpose from the matlab var
% analysis.maeDOIm_val	7,24E-06
%  ..
%  analysis.r2DOIh2_tst	0,962415516

% predefine the ranges as the ones which are physical. look into the plots
% shown in thesis
% analysis.rmseIMEP_val = errperf(1e-5 * yval(:,1)', 1e-5 * IMEP_cycle_hat, 'rmse');
% analysis.target_range_IMEP_phys = 10;
% analysis.target_range_IMEP_train = max(1e-5 * ytrain(:,1)) - min(1e-5 * ytrain(:,1));
% analysis.nrmseIMEP_val_trainData = (analysis.rmseIMEP_val / analysis.target_range_IMEP_train) * 100;
% analysis.nrmseIMEP_val_physData = (analysis.rmseIMEP_val / analysis.target_range_IMEP_phys) * 100;
% 
% analysis.rmseSOOT_val = errperf(yval(:,3)', SOOT_cycle_hat, 'rmse');
% analysis.target_range_SOOT_phys = 4;
% analysis.target_range_SOOT_train = max(ytrain(:,3)) - min(ytrain(:,3));
% analysis.nrmseSOOT_val_trainData = (analysis.rmseSOOT_val / analysis.target_range_SOOT_train) * 100;
% analysis.nrmseSOOT_val_physData = (analysis.rmseSOOT_val / analysis.target_range_SOOT_phys) * 100;


% === IMEP (scaled) ===
analysis.maeIMEP_val = errperf(1e-5 * yval(:,1)', 1e-5 * IMEP_cycle_hat, 'mae');
analysis.mseIMEP_val = errperf(1e-5 * yval(:,1)', 1e-5 * IMEP_cycle_hat, 'mse');
analysis.rmseIMEP_val = errperf(1e-5 * yval(:,1)', 1e-5 * IMEP_cycle_hat, 'rmse');
analysis.mspeIMEP_val = errperf(1e-5 * yval(:,1)', 1e-5 * IMEP_cycle_hat, 'mspe');
analysis.rmspeIMEP_val = errperf(1e-5 * yval(:,1)', 1e-5 * IMEP_cycle_hat, 'rmspe');
analysis.target_range_IMEP_val = max(1e-5 * ytrain(:,1)) - min(1e-5 * ytrain(:,1));
analysis.nrmseIMEP_val = (analysis.rmseIMEP_val / analysis.target_range_IMEP_val) * 100;
analysis.std_IMEP_val = std(1e-5 * ytrain(:,1)');
analysis.target_range_IMEP_sd4_val = 4*std(1e-5 * ytrain(:,1)');
analysis.target_range_IMEP_sd6_val = 6*std(1e-5 * ytrain(:,1)');
analysis.nrmse_sd6_IMEP_val = analysis.rmseIMEP_val / (6*analysis.std_IMEP_val) * 100;
analysis.nrmse_sd4_IMEP_val = analysis.rmseIMEP_val / (4*analysis.std_IMEP_val) * 100;
lower_p = prctile(1e-5 * ytrain(:,1)', 1); upper_p = prctile(1e-5 * ytrain(:,1)', 99);
analysis.range_clipped_IMEP_val = upper_p - lower_p;
analysis.nrmse_clipped_IMEP_val = analysis.rmseIMEP_val / analysis.range_clipped_IMEP_val * 100;
analysis.nmaeIMEP_val = (analysis.maeIMEP_val / analysis.target_range_IMEP_val) * 100;
analysis.nmae_sd4_IMEP_val = analysis.maeIMEP_val / (4 * analysis.std_IMEP_val) * 100;
analysis.nmae_sd6_IMEP_val = analysis.maeIMEP_val / (6 * analysis.std_IMEP_val) * 100;
analysis.nmae_clipped_IMEP_val = analysis.maeIMEP_val / analysis.range_clipped_IMEP_val * 100;


% === NOx ===
analysis.maeNOx_val = errperf(yval(:,2)', NOx_cycle_hat, 'mae');
analysis.mseNOx_val = errperf(yval(:,2)', NOx_cycle_hat, 'mse');
analysis.rmseNOx_val = errperf(yval(:,2)', NOx_cycle_hat, 'rmse');
analysis.mspeNOx_val = errperf(yval(:,2)', NOx_cycle_hat, 'mspe');
analysis.rmspeNOx_val = errperf(yval(:,2)', NOx_cycle_hat, 'rmspe');
analysis.target_range_NOx_val = max(ytrain(:,2)) - min(ytrain(:,2));
analysis.nrmseNOx_val = (analysis.rmseNOx_val / analysis.target_range_NOx_val) * 100;
analysis.std_NOx_val = std(ytrain(:,2)');
analysis.nrmse_sd4_NOx_val = analysis.rmseNOx_val / (4* analysis.std_NOx_val) * 100;
analysis.nrmse_sd6_NOx_val = analysis.rmseNOx_val / (6* analysis.std_NOx_val) * 100;
lower_p = prctile(ytrain(:,2)', 1); upper_p = prctile(ytrain(:,2)', 99); 
analysis.range_clipped_NOx_val = upper_p - lower_p;
analysis.nrmse_clipped_NOx_val = analysis.rmseNOx_val / analysis.range_clipped_NOx_val * 100;
analysis.nmaeNOx_val = (analysis.maeNOx_val / analysis.target_range_NOx_val) * 100;
analysis.nmae_sd4_NOx_val = analysis.maeNOx_val / (4 * analysis.std_NOx_val) * 100;
analysis.nmae_sd6_NOx_val = analysis.maeNOx_val / (6 * analysis.std_NOx_val) * 100;
analysis.nmae_clipped_NOx_val = analysis.maeNOx_val / analysis.range_clipped_NOx_val * 100;



% === SOOT ===
analysis.maeSOOT_val = errperf(yval(:,3)', SOOT_cycle_hat, 'mae');
analysis.mseSOOT_val = errperf(yval(:,3)', SOOT_cycle_hat, 'mse');
analysis.rmseSOOT_val = errperf(yval(:,3)', SOOT_cycle_hat, 'rmse');
% analysis.mspeSOOT_val = errperf(yval(:,3)', SOOT_cycle_hat, 'mspe');
% analysis.rmspeSOOT_val = errperf(yval(:,3)', SOOT_cycle_hat, 'rmspe');
analysis.target_range_SOOT_val = max(ytrain(:,3)) - min(ytrain(:,3));
analysis.nrmseSOOT_val = (analysis.rmseSOOT_val / analysis.target_range_SOOT_val) * 100;
analysis.std_SOOT_val = std(ytrain(:,3)');
analysis.target_range_SOOT_sd4_val = 4*std(ytrain(:,3)');
analysis.target_range_SOOT_sd6_val = 6*std(ytrain(:,3)');
analysis.nrmse_sd6_SOOT_val = analysis.rmseSOOT_val / (6*analysis.std_SOOT_val) * 100;
analysis.nrmse_sd4_SOOT_val = analysis.rmseSOOT_val / (4*analysis.std_SOOT_val) * 100;
% ~68% of data lies within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ
lower_p = prctile(ytrain(:,3)', 1); upper_p = prctile(ytrain(:,3)', 99); 
analysis.range_clipped_SOOT_val = upper_p - lower_p;
analysis.nrmse_clipped_SOOT_val = analysis.rmseSOOT_val / analysis.range_clipped_SOOT_val * 100;
analysis.nmaeSOOT_val = (analysis.maeSOOT_val / analysis.target_range_SOOT_val) * 100;
analysis.nmae_sd4_SOOT_val = analysis.maeSOOT_val / (4 * analysis.std_SOOT_val) * 100;
analysis.nmae_sd6_SOOT_val = analysis.maeSOOT_val / (6 * analysis.std_SOOT_val) * 100;
analysis.nmae_clipped_SOOT_val = analysis.maeSOOT_val / analysis.range_clipped_SOOT_val * 100;


% === MPRR (scaled) ===
analysis.maeMPRR_val = errperf(1e-5 * yval(:,4)', 1e-5 * MPRR_cycle_hat, 'mae');
analysis.mseMPRR_val = errperf(1e-5 * yval(:,4)', 1e-5 * MPRR_cycle_hat, 'mse');
analysis.rmseMPRR_val = errperf(1e-5 * yval(:,4)', 1e-5 * MPRR_cycle_hat, 'rmse');
analysis.mspeMPRR_val = errperf(1e-5 * yval(:,4)', 1e-5 * MPRR_cycle_hat, 'mspe');
analysis.rmspeMPRR_val = errperf(1e-5 * yval(:,4)', 1e-5 * MPRR_cycle_hat, 'rmspe');
analysis.target_range_MPRR_val = max(1e-5 * ytrain(:,4)) - min(1e-5 * ytrain(:,4));
analysis.nrmseMPRR_val = (analysis.rmseMPRR_val / analysis.target_range_MPRR_val) * 100;
analysis.std_MPRR_val = std(1e-5 *ytrain(:,4)');
analysis.nrmse_sd4_MPRR_val = analysis.rmseMPRR_val / (4*analysis.std_MPRR_val) * 100;
analysis.nrmse_sd6_MPRR_val = analysis.rmseMPRR_val / (6*analysis.std_MPRR_val) * 100;
% ~68% of data lies within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ
lower_p = prctile(1e-5 *ytrain(:,4)', 1); upper_p = prctile(1e-5 *ytrain(:,4)', 99); 
analysis.range_clipped_MPRR_val = upper_p - lower_p;
analysis.nrmse_clipped_MPRR_val = analysis.rmseMPRR_val / analysis.range_clipped_MPRR_val * 100;
analysis.nmaeMPRR_val = (analysis.maeMPRR_val / analysis.target_range_MPRR_val) * 100;
analysis.nmae_sd4_MPRR_val = analysis.maeMPRR_val / (4 * analysis.std_MPRR_val) * 100;
analysis.nmae_sd6_MPRR_val = analysis.maeMPRR_val / (6 * analysis.std_MPRR_val) * 100;
analysis.nmae_clipped_MPRR_val = analysis.maeMPRR_val / analysis.range_clipped_MPRR_val * 100;


clear mean;
% rmseIMEP_val = rmse((1e-5*yval(:,1))',(1e-5*IMEP_cycle_hat),"all"); % bar
% target_range_IMEP_val = max(yval(:,1)*1e-5) - min(yval(:,1)*1e-5);
% rmspeIMEP_val = ((rmseIMEP_val / target_range_IMEP_val)) * 100;
% rmseNOx_val=rmse(yval(:,2)',NOx_cycle_hat,"all");
% target_range_NOx_val = max(yval(:,2)) - min(yval(:,2));
% rmspeNOx_val = ((rmseNOx_val / target_range_NOx_val)) * 100;
% rmseSOOT_val=rmse(yval(:,3)', SOOT_cycle_hat,"all");
% target_range_SOOT_val = max(yval(:,3)) - min(yval(:,3));
% rmspeSOOT_val = ((rmseSOOT_val / target_range_SOOT_val)) * 100;
% rmseMPRR_val=rmse((1e-5*yval(:,4)),MPRR_cycle_hat*1e-5,"all");
% target_range_MPRR_val = max((1e-5*yval(:,4))) - min((1e-5*yval(:,4)));
% rmspeMPRR_val = ((rmseMPRR_val / target_range_MPRR_val)) * 100;

SSR_imep_val = sum((yval(:,1)' - IMEP_cycle_hat).^2); % Sum of squared residuals
TSS_imep_val = sum(((IMEP_cycle_hat - mean(IMEP_cycle_hat)).^2)); % Total sum of squares
analysis.r2IMEP_val = 1 - SSR_imep_val/TSS_imep_val; % R squared
SSR_NOx_val = sum((yval(:,2)' - NOx_cycle_hat).^2); 
TSS_NOx_val = sum(((NOx_cycle_hat - mean(NOx_cycle_hat)).^2));
analysis.r2NOx_val = 1 - SSR_NOx_val/TSS_NOx_val;
SSR_SOOT_val = sum((yval(:,3)' - SOOT_cycle_hat).^2); 
TSS_SOOT_val = sum(((SOOT_cycle_hat - mean(SOOT_cycle_hat)).^2));
analysis.r2SOOT_val = 1 - SSR_SOOT_val/TSS_SOOT_val;
SSR_MPRR_val = sum((yval(:,4)' - MPRR_cycle_hat).^2); 
TSS_MPRR_val = sum(((MPRR_cycle_hat - mean(MPRR_cycle_hat)).^2));
analysis.r2MPRR_val = 1 - SSR_MPRR_val/TSS_MPRR_val;


%% Scatter plot measured vs. predicted (linear ideally) VAL
figure
set(gcf,'color','w');
set(gcf,'units','points','position',[78.6,69,838.8,691.2])

subplot(2,2,1)
scatter(IMEP_cycle_hat * 1e-5, yval(:,1) * 1e-5, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(IMEP_cycle_hat) * 1e-5, 0);
max_scale_y = round(max(yval(:,1)) * 1e-5, 0);
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        xlabel('True IMEP / bar','Interpreter','latex')
        ylabel('Predicted IMEP / bar','Interpreter','latex')
end
% xlabel({'True IMEP / bar'},'Interpreter','latex')
% ylabel({'Predicted IMEP / bar'},'Interpreter','latex')
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f bar', analysis.rmseIMEP_val); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2IMEP_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,2)
scatter(NOx_cycle_hat, yval(:,2), 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(NOx_cycle_hat), -2);
max_scale_y = round(max(yval(:,2)), -2);
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        xlabel({'True NO$_x$ / ppm'},'Interpreter','latex')
        ylabel({'Predicted NO$_x$ / ppm'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f ppm', analysis.rmseNOx_val); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2NOx_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,3)
scatter(SOOT_cycle_hat, yval(:,3), 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = max(SOOT_cycle_hat);
max_scale_y = max(yval(:,3));
acc = 0.5; max_scale_x = round(max_scale_x/acc)*acc - 0.5; max_scale_y = round(max_scale_y/acc)*acc - 0.5; % MANIP
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        xlabel({'True Soot / (mg/m$^3$)'},'Interpreter','latex')
        ylabel({'Predicted Soot / (mg/m$^3$)'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f (mg/m$^3$)', analysis.rmseSOOT_val); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2SOOT_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');


subplot(2,2,4)
scatter(MPRR_cycle_hat*1e-5, yval(:,4)*1e-5, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(MPRR_cycle_hat*1e-5), -1);
max_scale_y = round(max(yval(:,4)*1e-5), -1);
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        xlabel({'True MPRR / (bar/CAD)'},'Interpreter','latex')
		ylabel({'Predicted MPRR / (bar/0.1CAR)'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f (bar/CAD)', analysis.rmseMPRR_val); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2MPRR_val); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

if save_plots_sw
    type = "/Prediction_Actual_Val"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50)
end

figure
set(gcf,'color','w');
figFileName="../Plots/"+ sprintf("%04d",MP)+'/'+ sprintf('%04d',trainingrun)+"/Training_Results_Val";
set(gcf,'units','points','position',[200,200,900,400])

subplot(4,1,1)
plot(yval(:,1) * 1e-5, 'r--')
hold on
plot(IMEP_cycle_hat(1:end) * 1e-5,'k-')
grid on
% title('Prediction on Validation Data')
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP / bar','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:2500:15000;
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')

subplot(4,1,2)
set(gcf,'units','points','position',[200,200,900,400])
plot(yval(:,2), 'r--')
hold on
plot(NOx_cycle_hat(1:end),'k-')         
grid on
ylim([0,1500])
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ \\ / ppm')},'Interpreter','latex')
    else
        ylabel({'NO$_x$';'/ ppm'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        ylabel('NO$_x$ / ppm','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:2500:15000;
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')

subplot(4,1,3)
set(gcf,'units','points','position',[200,200,900,400])
plot(yval(:,3), 'r--')
grid on
hold on
plot(SOOT_cycle_hat(1:end),'k-')
ylim([0,1.5])
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ \\ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel({'Soot';'/ (mg/m$^3$)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:2500:15000;
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')

subplot(4,1,4)
set(gcf,'units','points','position',[200,200,900,400])
plot(yval(:,4)*1e-5, 'r--')
hold on
plot(MPRR_cycle_hat(1:end)*1e-5,'k-')
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
ylim([0,50])
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ \\ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel({'MPRR';'/ (bar/CAD)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel('MPRR / (bar/CAD)','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:2500:15000;
set(gca,'fontsize',Opts.fontsize)
set(gca,'TickLabelInterpreter','latex')
ax.XRuler.Exponent = 0; % ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')

if save_plots_sw
    type = "/Prediction_Time_Val"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end

end

if plot_pred_test
%% Prediction on test dataset
if no_fb
    y_hat_tst = predict(h2df_model,[utest_1'; utest_2'; utest_3'; utest_4']) ; % without IMEP
else
    y_hat_tst = predict(h2df_model,[utest_1'; utest_2'; utest_3'; utest_4'; utest_5']) ; % with IMEP
end

y1_hat_tst = y_hat_tst(1,:);
y2_hat_tst = y_hat_tst(2,:);
y3_hat_tst = y_hat_tst(3,:);
y4_hat_tst = y_hat_tst(4,:);

% Denormalize Predictions
IMEP_cycle_hat_tst = dataTraindeNormalize(y1_hat_tst,y1_min,y1_range);
NOx_cycle_hat_tst = dataTraindeNormalize(y2_hat_tst,y2_min,y2_range);
SOOT_cycle_hat_tst = dataTraindeNormalize(y3_hat_tst,y3_min,y3_range);
MPRR_cycle_hat_tst = dataTraindeNormalize(y4_hat_tst,y4_min,y4_range);

%% get rmses etc postprocessing TEST
clear mean;
% rmseIMEP_tst = rmse((1e-5*ytest(:,1))',(1e-5*IMEP_cycle_hat_tst),"all"); % bar
% target_range_IMEP_tst = max(ytest(:,1)*1e-5) - min(ytest(:,1)*1e-5);
% rmspeIMEP_tst = ((rmseIMEP_tst / target_range_IMEP_tst)) * 100;
% rmseNOx_tst=rmse(ytest(:,2)',NOx_cycle_hat_tst,"all");
% target_range_NOx_tst = max(ytest(:,2)) - min(ytest(:,2));
% rmspeNOx_tst = ((rmseNOx_tst / target_range_NOx_tst)) * 100;
% rmseSOOT_tst=rmse(ytest(:,3)', SOOT_cycle_hat_tst,"all");
% target_range_SOOT_tst = max(ytest(:,3)) - min(ytest(:,3));
% rmspeSOOT_tst = ((rmseSOOT_tst / target_range_SOOT_tst)) * 100;
% rmseMPRR_tst=rmse((1e-5*ytest(:,4)),MPRR_cycle_hat_tst*1e-5,"all");
% target_range_MPRR_tst = max((1e-5*ytest(:,4))) - min((1e-5*ytest(:,4)));
% rmspeMPRR_tst = ((rmseMPRR_tst / target_range_MPRR_tst)) * 100;

% === IMEP (scaled) ===
analysis.maeIMEP_test = errperf(1e-5 * ytest(:,1)', 1e-5 * IMEP_cycle_hat_tst, 'mae');
analysis.mseIMEP_test = errperf(1e-5 * ytest(:,1)', 1e-5 * IMEP_cycle_hat_tst, 'mse');
analysis.rmseIMEP_test = errperf(1e-5 * ytest(:,1)', 1e-5 * IMEP_cycle_hat_tst, 'rmse');
analysis.mspeIMEP_test = errperf(1e-5 * ytest(:,1)', 1e-5 * IMEP_cycle_hat_tst, 'mspe');
analysis.rmspeIMEP_test = errperf(1e-5 * ytest(:,1)', 1e-5 * IMEP_cycle_hat_tst, 'rmspe');
analysis.target_range_IMEP_test = max(1e-5 * ytrain(:,1)) - min(1e-5 * ytrain(:,1));
analysis.nrmseIMEP_test = (analysis.rmseIMEP_test / analysis.target_range_IMEP_test) * 100;
analysis.std_IMEP_test = std(1e-5 * ytrain(:,1)');
analysis.nrmse_sd4_IMEP_test = analysis.rmseIMEP_test / (4*analysis.std_IMEP_test) * 100;
analysis.nrmse_sd6_IMEP_test = analysis.rmseIMEP_test / (6*analysis.std_IMEP_test) * 100;
lower_p = prctile(1e-5 * ytrain(:,1)', 1); upper_p = prctile(1e-5 * ytrain(:,1)', 99);
analysis.range_clipped_IMEP_test = upper_p - lower_p;
analysis.nrmse_clipped_IMEP_test = analysis.rmseIMEP_test / analysis.range_clipped_IMEP_test * 100;
analysis.nmaeIMEP_test = (analysis.maeIMEP_test / analysis.target_range_IMEP_test) * 100;
analysis.nmae_sd4_IMEP_test = analysis.maeIMEP_test / (4 * analysis.std_IMEP_test) * 100;
analysis.nmae_sd6_IMEP_test = analysis.maeIMEP_test / (6 * analysis.std_IMEP_test) * 100;
analysis.nmae_clipped_IMEP_test = analysis.maeIMEP_test / analysis.range_clipped_IMEP_test * 100;

% === NOx ===
analysis.maeNOx_test = errperf(ytest(:,2)', NOx_cycle_hat_tst, 'mae');
analysis.mseNOx_test = errperf(ytest(:,2)', NOx_cycle_hat_tst, 'mse');
analysis.rmseNOx_test = errperf(ytest(:,2)', NOx_cycle_hat_tst, 'rmse');
analysis.mspeNOx_test = errperf(ytest(:,2)', NOx_cycle_hat_tst, 'mspe');
analysis.rmspeNOx_test = errperf(ytest(:,2)', NOx_cycle_hat_tst, 'rmspe');
analysis.target_range_NOx_test = max(ytrain(:,2)) - min(ytrain(:,2));
analysis.nrmseNOx_test = (analysis.rmseNOx_test / analysis.target_range_NOx_test) * 100;
analysis.std_NOx_test = std(ytrain(:,2)');
analysis.nrmse_sd4_NOx_test = analysis.rmseNOx_test / (4* analysis.std_NOx_test) * 100;
analysis.nrmse_sd6_NOx_test = analysis.rmseNOx_test / (6* analysis.std_NOx_test) * 100;
lower_p = prctile(ytrain(:,2)', 1); upper_p = prctile(ytrain(:,2)', 99); 
analysis.range_clipped_NOx_test = upper_p - lower_p;
analysis.nrmse_clipped_NOx_test = analysis.rmseNOx_test / analysis.range_clipped_NOx_test * 100;
analysis.nmaeNOx_test = (analysis.maeNOx_test / analysis.target_range_NOx_test) * 100;
analysis.nmae_sd4_NOx_test = analysis.maeNOx_test / (4 * analysis.std_NOx_test) * 100;
analysis.nmae_sd6_NOx_test = analysis.maeNOx_test / (6 * analysis.std_NOx_test) * 100;
analysis.nmae_clipped_NOx_test = analysis.maeNOx_test / analysis.range_clipped_NOx_test * 100;

% === SOOT ===
analysis.maeSOOT_test = errperf(ytest(:,3)', SOOT_cycle_hat_tst, 'mae');
analysis.mseSOOT_test = errperf(ytest(:,3)', SOOT_cycle_hat_tst, 'mse');
analysis.rmseSOOT_test = errperf(ytest(:,3)', SOOT_cycle_hat_tst, 'rmse');
% analysis.mspeSOOT_test = errperf(ytest(:,3)', SOOT_cycle_hat_tst, 'mspe');
% analysis.rmspeSOOT_test = errperf(ytest(:,3)', SOOT_cycle_hat_tst, 'rmspe');
analysis.target_range_SOOT_test = max(ytrain(:,3)) - min(ytrain(:,3));
analysis.nrmseSOOT_test = (analysis.rmseSOOT_test / analysis.target_range_SOOT_test) * 100;
analysis.std_SOOT_test = std(ytrain(:,3)');
analysis.nrmse_sd4_SOOT_test = analysis.rmseSOOT_test / (4*analysis.std_SOOT_test) * 100;
analysis.nrmse_sd6_SOOT_test = analysis.rmseSOOT_test / (6*analysis.std_SOOT_test) * 100;
% ~68% of data lies within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ
lower_p = prctile(ytrain(:,3)', 1); upper_p = prctile(ytrain(:,3)', 99);
analysis.range_clipped_SOOT_test = upper_p - lower_p;
analysis.nrmse_clipped_SOOT_test = analysis.rmseSOOT_test / analysis.range_clipped_SOOT_test * 100;
analysis.nmaeSOOT_test = (analysis.maeSOOT_test / analysis.target_range_SOOT_test) * 100;
analysis.nmae_sd4_SOOT_test = analysis.maeSOOT_test / (4 * analysis.std_SOOT_test) * 100;
analysis.nmae_sd6_SOOT_test = analysis.maeSOOT_test / (6 * analysis.std_SOOT_test) * 100;
analysis.nmae_clipped_SOOT_test = analysis.maeSOOT_test / analysis.range_clipped_SOOT_test * 100;

% === MPRR (scaled) ===
analysis.maeMPRR_test = errperf(1e-5 * ytest(:,4)', 1e-5 * MPRR_cycle_hat_tst, 'mae');
analysis.mseMPRR_test = errperf(1e-5 * ytest(:,4)', 1e-5 * MPRR_cycle_hat_tst, 'mse');
analysis.rmseMPRR_test = errperf(1e-5 * ytest(:,4)', 1e-5 * MPRR_cycle_hat_tst, 'rmse');
analysis.mspeMPRR_test = errperf(1e-5 * ytest(:,4)', 1e-5 * MPRR_cycle_hat_tst, 'mspe');
analysis.rmspeMPRR_test = errperf(1e-5 * ytest(:,4)', 1e-5 * MPRR_cycle_hat_tst, 'rmspe');
analysis.target_range_MPRR_test = max(1e-5 * ytrain(:,4)) - min(1e-5 * ytrain(:,4));
analysis.nrmseMPRR_test = (analysis.rmseMPRR_test / analysis.target_range_MPRR_test) * 100;
analysis.std_MPRR_test = std(1e-5 *ytrain(:,4)');
analysis.nrmse_sd4_MPRR_test = analysis.rmseMPRR_test / (4*analysis.std_MPRR_test) * 100;
analysis.nrmse_sd6_MPRR_test = analysis.rmseMPRR_test / (6*analysis.std_MPRR_test) * 100;
% ~68% of data lies within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ
lower_p = prctile(1e-5 *ytrain(:,4)', 1); upper_p = prctile(1e-5 *ytrain(:,4)', 99);
analysis.range_clipped_MPRR_test = upper_p - lower_p;
analysis.nrmse_clipped_MPRR_test = analysis.rmseMPRR_test / analysis.range_clipped_MPRR_test * 100;
analysis.nmaeMPRR_test = (analysis.maeMPRR_test / analysis.target_range_MPRR_test) * 100;
analysis.nmae_sd4_MPRR_test = analysis.maeMPRR_test / (4 * analysis.std_MPRR_test) * 100;
analysis.nmae_sd6_MPRR_test = analysis.maeMPRR_test / (6 * analysis.std_MPRR_test) * 100;
analysis.nmae_clipped_MPRR_test = analysis.maeMPRR_test / analysis.range_clipped_MPRR_test * 100;

SSR_imep_test = sum((ytest(:,1)' - IMEP_cycle_hat_tst).^2); % Sum of squared residuals
TSS_imep_test = sum(((IMEP_cycle_hat_tst - mean(IMEP_cycle_hat_tst)).^2)); % Total sum of squares
analysis.r2IMEP_test = 1 - SSR_imep_test/TSS_imep_test; % R squared
SSR_NOx_test = sum((ytest(:,2)' - NOx_cycle_hat_tst).^2); 
TSS_NOx_test = sum(((NOx_cycle_hat_tst - mean(NOx_cycle_hat_tst)).^2));
analysis.r2NOx_test = 1 - SSR_NOx_test/TSS_NOx_test;
SSR_SOOT_test = sum((ytest(:,3)' - SOOT_cycle_hat_tst).^2); 
TSS_SOOT_test = sum(((SOOT_cycle_hat_tst - mean(SOOT_cycle_hat_tst)).^2));
analysis.r2SOOT_test = 1 - SSR_SOOT_test/TSS_SOOT_test;
SSR_MPRR_test = sum((ytest(:,4)' - MPRR_cycle_hat_tst).^2); 
TSS_MPRR_test = sum(((MPRR_cycle_hat_tst - mean(MPRR_cycle_hat_tst)).^2));
analysis.r2MPRR_test = 1 - SSR_MPRR_test/TSS_MPRR_test;


%% Scatter plot measured vs. predicted (linear ideally) TEST
figure
set(gcf,'color','w');
set(gcf,'units','points','position',[78.6,69,838.8,691.2])

subplot(2,2,1)
scatter(IMEP_cycle_hat_tst * 1e-5, ytest(:,1) * 1e-5, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(IMEP_cycle_hat_tst) * 1e-5, 0);
max_scale_y = round(max(ytest(:,1)) * 1e-5, 0);
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        xlabel('True IMEP / bar','Interpreter','latex')
        ylabel('Predicted IMEP / bar','Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f bar', analysis.rmseIMEP_test); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2IMEP_test); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,2)
scatter(NOx_cycle_hat_tst, ytest(:,2), 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(NOx_cycle_hat_tst), -2);
max_scale_y = round(max(ytest(:,2)), -2);
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        xlabel({'True NO$_x$ / ppm'},'Interpreter','latex')
        ylabel({'Predicted NO$_x$ / ppm'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f ppm', analysis.rmseNOx_test); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2NOx_test); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

subplot(2,2,3)
scatter(SOOT_cycle_hat_tst, ytest(:,3), 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = max(SOOT_cycle_hat_tst);
max_scale_y = max(ytest(:,3));
acc = 0.5; max_scale_x = round(max_scale_x/acc)*acc - 0.5; max_scale_y = round(max_scale_y/acc)*acc - 0.5; % MANIP
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        xlabel({'True Soot / (mg/m$^3$)'},'Interpreter','latex')
        ylabel({'Predicted Soot / (mg/m$^3$)'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f (mg/m$^3$)', analysis.rmseSOOT_test); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2SOOT_test); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');


subplot(2,2,4)
scatter(MPRR_cycle_hat_tst*1e-5, ytest(:,4)*1e-5, 4, 'k', 'x', 'LineWidth', 4);
max_scale_x = round(max(MPRR_cycle_hat_tst*1e-5), -1);
max_scale_y = round(max(ytest(:,4)*1e-5), -1);
xlim([0, max(max_scale_x, max_scale_y)]); ylim([0, max(max_scale_x, max_scale_y)]);
line([0, max(max_scale_x, max_scale_y)], [0, max(max_scale_x, max_scale_y)], 'Color', 'blue', 'LineWidth', 1, 'DisplayName', 'Ideal Prediction');
grid on; box on;
if Opts.LatexLabels
        xlabel({strcat('True $ ',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
        ylabel({strcat('Predicted $ ',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        xlabel({'True MPRR / (bar/CAD)'},'Interpreter','latex')
		ylabel({'Predicted MPRR / (bar/0.1CAR)'},'Interpreter','latex')
end
set(gca,'FontSize',fntsze)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
pos = get(gca, 'Position');
textString1 = sprintf('RMSE: %.2f (bar/CAD)', analysis.rmseMPRR_test); % '%.2f' formats the double to 2 decimal places
textString2 = sprintf('R2: %.2f', analysis.r2MPRR_test); % '%.2f' formats the double to 2 decimal places
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2), 0.50*pos(3), 0.06*pos(4)], 'String', textString1, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex'); %  [x y w h]
annotation('textbox', [pos(1) + 0.5*pos(3), pos(2) + 0.06 * pos(4), 0.50*pos(3), 0.06*pos(4)], 'String', textString2, ...
    'FitBoxToText', 'off', 'BackgroundColor', 'white', 'FontSize', fntsze,'Interpreter','latex');

if save_plots_sw
    type = "/Prediction_Actual_Test"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50)
end

%% Plotting on test dataset
figure
set(gcf,'color','w');
set(gcf,'units','points','position',[200,200,900,400])

subplot(4,1,1)
plot(ytest(:,1) * 1e-5, 'r--')
hold on
plot(IMEP_cycle_hat_tst * 1e-5,'k-')
grid on
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ \\ / bar')},'Interpreter','latex')
    else
        ylabel({'IMEP';'/ bar'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_pme,'$ / bar')},'Interpreter','latex')
    else
        ylabel('IMEP / bar','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:1000:5000;
set(gca,'fontsize',Opts.fontsize)
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')

subplot(4,1,2)
set(gcf,'units','points','position',[200,200,900,400])
plot(ytest(:,2), 'r--')
hold on
plot(NOx_cycle_hat_tst,'k-')
grid on
ylim([0,1500])
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ \\ / ppm')},'Interpreter','latex')
    else
        ylabel({'NO$_x$';'/ ppm'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cnox,'$ / ppm')},'Interpreter','latex')
    else
        ylabel('NO$_x$ / ppm','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:1000:5000;
set(gca,'fontsize',Opts.fontsize)
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')

subplot(4,1,3)
set(gcf,'units','points','position',[200,200,900,400])
plot(ytest(:,3), 'r--')
hold on
plot(SOOT_cycle_hat_tst,'k-')
grid on
ylim([0,1.5])
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ \\ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel({'Soot';'/ (mg/m$^3$)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_cpm,'$ / (mg/m$^3$)')},'Interpreter','latex')
    else
        ylabel('Soot / (mg/m$^3$)','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:1000:5000;
set(gca,'fontsize',Opts.fontsize)
ax.XRuler.Exponent = 0; ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')


subplot(4,1,4)
set(gcf,'units','points','position',[200,200,900,400])
plot(ytest(:,4)* 1e-5, 'r--')
hold on
plot(MPRR_cycle_hat_tst* 1e-5,'k-')
grid on
xlabel("Cycles / -",'Interpreter', 'latex')
ylim([0,50])
if Opts.multi_lines_ylabel % 2 lines
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ \\ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel({'MPRR';'/ (bar/CAD)'},'Interpreter','latex')
    end
else % 1 line
    if Opts.LatexLabels
        ylabel({strcat('$',Opts.ltx_dpm,'$ / (bar/CAD)')},'Interpreter','latex')
    else
        ylabel('MPRR / (bar/CAD)','Interpreter','latex')
    end
end
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XTick = 0:1000:5000;
set(gca,'fontsize',Opts.fontsize)
ax.XRuler.Exponent = 0; % ax.XTickLabel = [];
% legend({'Measured','Predicted'},'Location','southeast','Orientation','horizontal')


if save_plots_sw
    type = "/Prediction_Time_Test"; 
    save_plots(gcf, MP, trainingrun, type, plot_step_zoom, 50, Opts.multi_lines_ylabel)
end

end


%% Define prediction function- we can use it later for MPC

% We have 8 hidden states 
unit_size = LSTMStateNum;
% Use aliases for network parameters

% Fully connected layers- 1
WFc1 =  h2df_model.Layers(2, 1).Weights;
bFc1 =  h2df_model.Layers(2, 1).Bias;

% Fully connected layers- 2
WFc2 =  h2df_model.Layers(4, 1).Weights;
bFc2 =  h2df_model.Layers(4, 1).Bias;

% Fully connected layers- 3
WFc3 =  h2df_model.Layers(6, 1).Weights;
bFc3 =  h2df_model.Layers(6, 1).Bias;


% Recurrent weights
Rr =  h2df_model.Layers(8, 1).RecurrentWeights(1:unit_size, :);
Rz=  h2df_model.Layers(8, 1).RecurrentWeights(unit_size+1:2*unit_size, :);
Rh =  h2df_model.Layers(8, 1).RecurrentWeights(2*unit_size+1:3*unit_size, :);
%Ro =  h2df_model.Layers(8, 1).RaecurrentWeights(3*unit_size+1:end, :);

% Input weights
wr =  h2df_model.Layers(8, 1).InputWeights(1:unit_size, :);
wz =  h2df_model.Layers(8, 1).InputWeights(unit_size+1:2*unit_size, :);
wh =  h2df_model.Layers(8, 1).InputWeights(2*unit_size+1:3*unit_size, :);
%wo =  h2df_model.Layers(8, 1).InputWeights(3*unit_size+1:end, :);

% Bias weights
br =  h2df_model.Layers(8, 1).Bias(1:unit_size, :);
bz =  h2df_model.Layers(8, 1).Bias(unit_size+1:2*unit_size, :);
bh =  h2df_model.Layers(8, 1).Bias(2*unit_size+1:3*unit_size, :);
%bo =  h2df_model.Layers(8, 1).Bias(3*unit_size+1:end, :);

% Fully connected layers- 4
WFc4 =  h2df_model.Layers(9, 1).Weights;
bFc4 =  h2df_model.Layers(9, 1).Bias;

% Fully connected layers- 5
WFc5 =  h2df_model.Layers(11, 1).Weights;
bFc5 =  h2df_model.Layers(11, 1).Bias;

% Fully connected layers- 6
WFc6 =  h2df_model.Layers(13, 1).Weights;
bFc6 =  h2df_model.Layers(13, 1).Bias;

%% Assigining parameters to a structure

Par.Rz = double(Rz);
Par.Rr = double(Rr);
Par.Rh = double(Rh);
%Par.Ro = double(Ro);
Par.wz = double(wz);
Par.wr = double(wr);
Par.wh = double(wh);
%Par.wo = double(wo);
Par.bz = double(bz);
Par.br = double(br);
Par.bh = double(bh);
%Par.bo = double(bo);
Par.WFc1 = double(WFc1);
Par.bFc1 = double(bFc1);
Par.WFc2 = double(WFc2);
Par.bFc2 = double(bFc2);
Par.WFc3 = double(WFc3);
Par.bFc3 = double(bFc3);
Par.WFc4 = double(WFc4);
Par.bFc4 = double(bFc4);
Par.WFc5 = double(WFc5);
Par.bFc5 = double(bFc5);
Par.WFc6 = double(WFc6);
Par.bFc6 = double(bFc6);
Par.nCellStates = 0; %change for lstm to hiddenstates (Par.nHiddenStates
Par.nHiddenStates = unit_size;
Par.nStates = Par.nHiddenStates;
Par.nActions = featureDimension;
Par.nOutputs = numResponses; 
Par.TotalLearnables = analysis.TotalLearnables(run_nmbr,1) ;
Par.FinalRMSE = analysis.FinalRMSE(run_nmbr,1);
Par.FinalValidationLoss = analysis.FinalValidationLoss(run_nmbr,1);
Par.ElapsedTime = analysis.ElapsedTime(run_nmbr,1);
Par.Savename = analysis.savename(run_nmbr,1);
% Par.RMSE_Test = [rmseIMEP_tst, rmseNOx_tst, rmseSOOT_tst, rmseMPRR_tst];
% Par.RMSE_Val = [rmseIMEP_val, rmseNOx_val, rmseSOOT_val, rmseMPRR_val];
% Par.RMSPE_Test = [rmspeIMEP_tst, rmspeNOx_tst, rmspeSOOT_tst, rmspeMPRR_tst];
% Par.RMSPE_Val = [rmspeIMEP_val, rmspeNOx_val, rmspeSOOT_val, rmspeMPRR_val];
% Par.R2_Test = [r2IMEP_test, r2NOx_test, r2SOOT_test, r2MPRR_test];
% Par.R2_Val = [r2IMEP_val, r2NOx_val, r2SOOT_val, r2MPRR_val];

if save_analysis == true
    save(['../Results/Analysis_',savename],"analysis")
end

if overwrite_par == true || do_training == true
    save(['../Results/Par_',savename],"Par")
end

trainingrun = trainingrun + 1; % increase when doing grid search

if break_loop break
end

end

if break_loop break
end

end

%% verfiy own function
if verify_my_func

% %% Simulating your model
% 
% 
if no_fb
    uts = [uval_1';uval_2';uval_3';uval_4'];
else
    uts = [uval_1';uval_2';uval_3';uval_4';uval_5'];
end
xt1 = zeros(Par.nStates,1);
y1_hat_myfnc = zeros(length(uval_1),1);
y2_hat_myfnc = zeros(length(uval_1),1);
y3_hat_myfnc = zeros(length(uval_1),1);
y4_hat_myfnc = zeros(length(uval_1),1);

for i = 1:length(uval_1)

[xt,y] = MyGRUstateFnc(xt1, uts(:,i),Par);

y1_hat_myfnc(i,1) = y(1);
y2_hat_myfnc(i,1) = y(2);
y3_hat_myfnc(i,1) = y(3);
y4_hat_myfnc(i,1) = y(4);
xt1 = xt;


end


figure(15) % comparing mylstmstatefun with matlab prediction - this should be the same!
set(gcf, 'Position', [100, 100, 900, 1200]);
set(gcf,'color','w');


subplot(4,1,1)
plot(y1_hat)
hold on
plot(y1_hat_myfnc, 'r--')
grid on
ylabel('IMEP','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted IMEP model','Predicted IMEP function'},'Location','southeast','Orientation','horizontal')


subplot(4,1,2)
plot(y2_hat)
hold on
plot(y2_hat_myfnc, 'r--')
grid on
ylabel('NOx','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted NOx model','Predicted NOx function'},'Location','southeast','Orientation','horizontal')

subplot(4,1,3)
plot(y3_hat)
hold on
plot(y3_hat_myfnc, 'r--')
grid on
ylabel('Soot','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted Soot model','Predicted Soot function'},'Location','southeast','Orientation','horizontal')

subplot(4,1,4)
plot(y4_hat)
hold on
plot(y4_hat_myfnc, 'r--')
grid on
ylabel('MPRR','Interpreter','latex')
set(gca,'FontSize',14)
set(gca,'TickLabelInterpreter','latex')
ax = gca;
ax.XRuler.Exponent = 0;
legend({'Predicted MPRR model','Predicted MPRR function'},'Location','southeast','Orientation','horizontal')
end

%% EXTERNAL FUNCTIONS
function save_plots(gcf, MP, trainingrun, type, zoom, resolution, multi_lines_ylabel, externalize, yyxais_right)
    if (~exist('resolution', 'var'))
        resolution = 150;
    end
    if (~exist('externalize', 'var'))
        externalize = false;
    end
    if (~exist('zoom', 'var'))
        zoom = false;
    end
    if (~exist('yyxais_right', 'var'))
        yyxais_right = false;
    end
    if (~exist('multi_lines_ylabel', 'var'))
        multi_lines_ylabel = false;
    end

    if zoom
        figFileName="../Plots/"+ sprintf("%04d",MP)+'/'+ sprintf('%04d',trainingrun) + type + '_zoom';
    else
        figFileName="../Plots/"+ sprintf("%04d",MP)+'/'+ sprintf('%04d',trainingrun) + type;
    end
    savefig(figFileName);
    saveas(gcf,figFileName,"jpg");
    % saveas(gcf,figFileName,"epsc");
    % saveas(gcf,figFileName,"pdf");
    if resolution > 0
        cleanfigure('targetResolution', resolution)
    end
    if multi_lines_ylabel
        axis_code = 'ylabel style={align=center},scaled ticks=false,';
    else
        axis_code = 'scaled ticks=false,';
    end
    if yyxais_right
        axis_code = 'ylabel style={align=center}, axis y line*=right, every outer x axis line/.append style={draw=none},every x tick label/.append style={text opacity=0},every x tick/.append style={draw=none},';
    end
    if externalize
        matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions',axis_code, 'externalData',true);
    else
        matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions', axis_code);
    end
    % axis_code = 'yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, scaled ticks=false,';
    % axis_code = 'yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, scaled ticks=false,';
    % matlab2tikz(convertStringsToChars(figFileName+'.tex'),'showInfo', false, 'width','\figW','height','\figH','extraAxisOptions',axis_code); % 'externalData',true
    
    %export_fig(figFileName,'-eps');

end