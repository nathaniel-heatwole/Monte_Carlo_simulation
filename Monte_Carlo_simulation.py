# MONTE_CARLO_SIMULATION.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses Monte Carlo simulation to test the central limit theorem, by averaging random draws from several constituent distributions over many trials
# Training data are synthetic and randomly generated (triangular distributions)

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
np.random.seed(1234)

#--------------#
#  PARAMETERS  #
#--------------#

total_dists = 10      # total constituent distributions (each triangular)
dist_width = 10       # width of each constituent distribution (along x-axis)
obs_per_dist = 10**6  # total obs to randomly generate from each constituent distribution
x_start = 0           # leftmost point of leftmost constituent distribution (x-value)

#-----------------#
#  DISTRIBUTIONS  #
#-----------------#

# parameters for first constituent distribution (triangular)
left_val = x_start
right_val = left_val + dist_width
mode_val = left_val
tri_params = [[left_val, mode_val, right_val]]

# parameters for all other constituent distributions (triangular)
for d in range(total_dists - 1):
    left_val += dist_width
    right_val += dist_width
    # put the mode alternatively at the minimum or maximum value
    if (d % 2) == 1:
        mode_val = left_val
    else:
        mode_val = right_val
    tri_params.append([left_val, mode_val, right_val])

x_end = right_val

#--------------#
#  SIMULATION  #
#--------------#

# generate constituent distributions (random draws)
dists = pd.DataFrame()
for d in range(total_dists):
    left_val = tri_params[d][0]
    mode_val = tri_params[d][1]
    right_val = tri_params[d][2]
    dists['x' + str(d + 1)] = np.random.triangular(left=left_val, mode=mode_val, right=right_val, size=obs_per_dist)  # triangular distributions

# performs Monte Carlo simulation, by combining random draws from all distributions (rowwise) myriad different times (rows)
dists['x avg'] = dists.mean(axis=1)  # average constituent distributions (rowwise)
dists.sort_values(by=['x avg'], axis=0, ignore_index=True, inplace=True)
dists['cdf'] = (dists.index + 1) / len(dists)  # empirical cdf (central limit theorem predicts this will be asymptotically normally distributed)

#---------#
#  PLOTS  #
#---------#

# parameters
title_size = 10
axis_labels_size = 8
axis_ticks_size = 7
legend_size = 6
line_width = 1
y_margin = 0.015
histo_alpha = 0.5
histo_width = 0.15
histo_bins_per_dist = 15
nobs_scatter_matrix = int(0.01*obs_per_dist)  # total obs for scatter matrix plots (prevents them from being overly cluttered)

color_list = ['blue', 'darkorange', 'green', 'brown', 'purple', 'red', 'pink', 'cyan', 'olive', 'gray']

df_pairplot = dists.drop(columns=['cdf', 'x avg'], axis=1)
df_pairplot = df_pairplot.sample(n=nobs_scatter_matrix)

mu = np.mean(dists['x avg'])
sigma = np.std(dists['x avg'])
normal_dist_text = 'Normal(μ=' + str(f'{mu:.2f}') + ', σ=' + str(f'{sigma:.2f}') + ')'

# scatter matrix (black = univariate histograms, blue = 2D relationships)
scatter_matrix_title = 'Monte Carlo Simulation - Scatter Matrix (triangular distributions)'
sns.set_style('darkgrid')
sns.set(rc={'figure.facecolor':'lightblue'})
s1 = sns.pairplot(df_pairplot, diag_kind='hist', markers='.', diag_kws={'color':'black'})
s1.fig.suptitle(scatter_matrix_title, y=1.03, fontsize=50, fontweight='bold')
s1.set(xlabel='', ylabel='', xticklabels=[], yticklabels=[])
plt.show(True)

# constituent distributions
fig1, ax = plt.subplots(facecolor='lightblue')
plt.gca().set_facecolor('white')
plt.title('Monte Carlo Simulation - Constituent Distributions (triangular)', fontsize=title_size, fontweight='bold')
max_bin_cnt = 0
for d in range(total_dists):
    d2 = d if d < len(color_list) else d % len(color_list)
    h = plt.hist(dists['x' + str(d+1)], bins=histo_bins_per_dist, label='D' + str(d+1), color=color_list[d2], edgecolor='black', alpha=histo_alpha, linewidth=histo_width)
    if np.max(h[0]) > max_bin_cnt:
        max_bin_cnt = np.max(h[0])
plt.legend(loc='upper center', ncol=int(0.5*total_dists), fontsize=legend_size, facecolor='whitesmoke', framealpha=1)
plt.xlabel('X', fontsize=axis_labels_size)
plt.ylabel('Count (bin)', fontsize=axis_labels_size)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xlim(x_start, x_end)
plt.ylim(0, 0.16*obs_per_dist)
plt.xticks(np.arange(x_start, x_end + dist_width, dist_width), fontsize=axis_ticks_size)
plt.yticks(fontsize=axis_ticks_size)
plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
plt.show(True)

xmin = np.floor(min(dists['x avg']))
xmax = np.ceil(max(dists['x avg']))

# central limit theorem (average)
fig2, ax1 = plt.subplots(facecolor='lightblue')
plt.gca().set_facecolor('white')
plt.title('Monte Carlo Simulation - Central Limit Theorem', fontsize=title_size, fontweight='bold')
dists.plot(y='cdf', x='x avg', ax=ax1, color='red', linewidth=line_width, legend=True, label='Probability (cumulative)', zorder=30)
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.995), fontsize=legend_size, facecolor='whitesmoke', framealpha=1).set_zorder(30)
plt.text(xmax - 0.01*(xmax - xmin), 0, normal_dist_text, va='bottom', ha='right', fontsize=legend_size, zorder=30)
plt.xlabel('Average value', fontsize=axis_labels_size)
plt.ylabel('Probability', fontsize=axis_labels_size)
plt.xticks(fontsize=axis_ticks_size)
plt.yticks(fontsize=axis_ticks_size)
plt.xlim(xmin, xmax)
plt.ylim(-y_margin, 1 + y_margin)
plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
ax2 = ax1.twinx()
sns.histplot(dists['x avg'], bins=int(obs_per_dist**(1/3)), alpha=histo_alpha, linewidth=histo_width, color='royalblue', edgecolor='black', ax=ax2, label='Count (bin)', zorder=5)
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.95), fontsize=legend_size, facecolor='whitesmoke', framealpha=1).set_zorder(30)
plt.ylabel('Count', fontsize=axis_labels_size)
plt.xticks(fontsize=axis_ticks_size)
plt.yticks(fontsize=axis_ticks_size)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.grid(True, color='lightgray', linewidth=0.75, alpha=0.5, zorder=0)
plt.show(True)

#----------------#
#  EXPORT PLOTS  #
#----------------#

s1.savefig('MC-figure-0.jpg', dpi=300)
pdf = PdfPages('Monte_Carlo_simulation_graphics.pdf')
fig_list = [fig1, fig2]
for fig in fig_list:
    f = fig_list.index(fig) + 1
    fig.savefig('MC-figure-' + str(f) + '.jpg', dpi=300)
    pdf.savefig(fig)
pdf.close()
del pdf, fig, f

###

# runtime
runtime_sec = round(time.time()-time0, 2)
if runtime_sec < 60:
    print('\nruntime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec/60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\nruntime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0

