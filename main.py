from utils.plotter import plot_results,plot_results2
from utils.parse_table import print_tables

import pandas as pd

# df  = pd.read_csv('dc_sameItem.csv')
# df2  = pd.read_csv('dc_BERT.csv')

# print(df.head)

# plot_results('#content#finetune#0$finetune$IR(F)$bert#$SR(F)$LS(F)$HN(F)$SQ(F)$SQS(F)$PC(F)$Mon#Jun#20#09:16:51#2022#LM','dot',df)
# plot_results('bert#base#uncased','dot',df)

# plot_results2(df2,df)
# plot_results2(df2,df,'MAP')


print_tables('every_k.csv')
