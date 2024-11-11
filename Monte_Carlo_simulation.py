# MONTE_CARLO_SIMULATION.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses Monte Carlo simulation to test the central limit theorem, by averaging random draws from several constituent distributions over myriad trials
# Input data: synthetic and randomly generated (triangular distributions)

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
np.random.seed(1234)
ver = ''  # version (empty or integer)

topic = 'Monte Carlo simulation'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

total_dists = 10       # total constituent distributions (triangular)
dist_width = 10        # width of each constituent distribution (along x-axis)
x_start = 0            # leftmost point of leftmost constituent distribution (x-value)
obs_per_dist = 10**6   # total obs to randomly generate from each constituent distribution

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
title_size = 11
axis_labels_size = 8
axis_ticks_size = 8
legend_size = 8
histogram_bins_per_dist = 15
nobs_scatter_matrix = 10**3      # total obs for scatter matrix plots (prevents them from being overly cluttered)
y_margin_title_pairplot = 1.03   # space for title atop pairplot (one = no margin)
y_margin_label_dists_plot = 1.2  # space for legend atop constituent distributions plot (one = no margin)

# scatter matrix (black = univariate histograms, blue = 2D relationships)
df_pairplot = dists.drop(columns=['cdf', 'x avg'], axis=1)
df_pairplot = df_pairplot.head(n=int(nobs_scatter_matrix))
s1 = sns.pairplot(df_pairplot, diag_kind='hist', markers='.', diag_kws={'color':'black'})
scatter_matrix_title = 'Scatter plot matrix (' + str(total_dists) + ' triangular distributions) (' + str(int(nobs_scatter_matrix)) + ' obs each)'
s1.fig.suptitle(scatter_matrix_title, y=y_margin_title_pairplot, fontsize=50, fontweight='bold')
s1.set(xlabel='', ylabel='', xticklabels=[], yticklabels=[])
sns.set_style('darkgrid')
plt.show(True)

# constituent distributions (all together)
fig1, ax = plt.subplots()
plt.title(topic + ' - constituent distributions (' + str(total_dists) + ' triangular)', fontsize=title_size, fontweight='bold')
max_bin_cnt = 0
for d in range(total_dists):
    h = plt.hist(dists['x' + str(d + 1)], bins=histogram_bins_per_dist, label='d' + str(d + 1), alpha=0.8, linewidth=0.5)
    if np.max(h[0]) > max_bin_cnt:
        max_bin_cnt = np.max(h[0])
plt.legend(loc='upper center', ncol=int(total_dists / 2), fontsize=legend_size, facecolor='white', framealpha=1)
plt.xlabel('X', fontsize=axis_labels_size)
plt.ylabel('Count (bin)', fontsize=axis_labels_size)
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.ylim(0, int(y_margin_label_dists_plot * max_bin_cnt))
plt.xticks(np.arange(x_start, x_end + dist_width, dist_width), fontsize=axis_ticks_size)
plt.yticks(fontsize=axis_ticks_size)
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

# average value (histogram, empirical cdf)
fig2, ax1 = plt.subplots()
plt.title(topic + ' - central limit theorem', fontsize=title_size, fontweight='bold')
dists.plot(y='cdf', x='x avg', ax=ax1, color='red', legend=True, label='cumulative probability', zorder=10)
plt.legend(loc='upper left', fontsize=legend_size, facecolor='white', framealpha=1)
plt.xlabel('Average value', fontsize=axis_labels_size)
plt.ylabel('Probability', fontsize=axis_labels_size)
plt.xticks(fontsize=axis_ticks_size)
plt.yticks(fontsize=axis_ticks_size)
ax2 = ax1.twinx()
sns.histplot(dists['x avg'], bins=int(obs_per_dist**(1/3)), alpha=0.5, ax=ax2, label='count', zorder=5)
plt.legend(loc='lower left', bbox_to_anchor=(0, 0.85), fontsize=legend_size, facecolor='white', framealpha=1)
plt.ylabel('Count (bin)', fontsize=axis_labels_size)
plt.yticks(fontsize=axis_ticks_size)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

#----------#
#  EXPORT  #
#----------#

# export plots (pdf, png)
file_title = topic_underscore + '_plots' + ver
s1.savefig(file_title + '.png')
pdf = PdfPages(file_title + '.pdf')
for f in [fig1, fig2]:
    pdf.savefig(f)
pdf.close()
del pdf, f

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


