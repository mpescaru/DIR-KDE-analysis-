import percentage_calc as per
import dir_auc as auc


bandwidth = 0.01
n_iter=500



#per.compute_percent_iso('~/syllable_data__2025-03-19.xlsx')
#per.percent_iso_music('/Users/maria/Desktop/sakata_lab/dir_analysis/tabular_music_data.csv', n_iter = 500)
#auc.plot_music_auc_curves('/Users/maria/Desktop/sakata_lab/dir_analysis/tabular_music_data.csv', n_iter = 500)
auc.plot_bird_auc_curves('/Users/maria/Desktop/sakata_lab/dir_analysis/syllable_data__2025-03-19.xlsx')



