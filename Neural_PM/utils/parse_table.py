import pandas as pd

import numpy as np


def format(v):
    if '±' in str(v):
        sp = v.split('±')
        # a = float(sp[0]) * 100
        a = float(sp[0])
        # b = float(sp[1]) * 100
        b = float(sp[1])
        a = '{:.3f}'.format(a)[1:]
        b = '{:.3f}'.format(b)[1:]
        return str(a) + ' ± ' + str(b)
    else:
        # return str('{:.2f}'.format(float(v) * 100))
        return str('{:.3f}'.format(float(v)))[1:]


def get_latex_table2(x, y, df,k):
    table = "\\begin{table*}[h] \centering \\begin{tabular}{@{}lll@{}} \\toprule \n"
    table += "\\textbf{Model}   &    \\textbf{R-Precision}          & \\textbf{MAP}             \\\\ \midrule \n"
    for i in range(x , x + 9):
        # print(df.iloc[i,y])
        table += "\\begin{tabular}[c]{@{}l@{}}" +str(df.iloc[i, y]) + " \end{tabular} " + " &   " + \
                 format(df.iloc[i, y + 1]) + "  & " + format(df.iloc[i, y + 2]) + " \\\\ \midrule \n"

    table += "\end{tabular} \n"
    table += "\caption{CPM-BERT (our base model) results on PMD dataset compared to conventional (TF-IDF) and current state-of-the-art self-supervised IR models with $k_R="+k+"$.} \n \label{2_"+k+"} \n \end{table*} \n"
    return table


def get_latex_table3(x, y, df,k):
    table = "\\begin{table*}[h] \centering \\begin{tabular}{@{}lllll@{}} \\toprule \n"
    table += "\\textbf{Model}       & \\textbf{Positive Setting}    & \\textbf{Negative Setting}  & \\textbf{R-Precision}   & \\textbf{MAP}    \\\\ \midrule \n"
    for i in range(x + 1, x + 21):
        table +=  str(df.iloc[i, y]) + "  & \\begin{tabular}[c]{@{}l@{}} " + str(df.iloc[i, y + 1]) + " \end{tabular}  & \\begin{tabular}[c]{@{}l@{}} " + \
                 str(df.iloc[i, y + 2]) + "  \end{tabular}  &  " + format(
            df.iloc[i, y + 3]) + "   &   " + format(df.iloc[i, y + 4]) + "    \\\\ \midrule \n"
    table += "\end{tabular} \n"
    table += "\caption{Exploring different techniques for CPM-BERT with $k_R="+k+"$ and 90\% confidence interval. Positive and negative sampling methods are explained in section \\ref{positive_sampling}}  \n \label{3_"+k+"}  \n \end{table*} \n"
    return table




def get_table_3col(x,y,df):
    table = "\\begin{table}[h] \centering \\begin{tabular}{@{}l|ll|ll|ll|@{}} \\toprule \n"
    table += "       & k=1   &    &  k=10   &   &    k=avg   &  \\\\ \\midrule"
    table += "\\textbf{Model}   &    \\textbf{R-Prec}          & \\textbf{MAP}    &    \\textbf{R-Prec}          & \\textbf{MAP}   &    \\textbf{R-Prec}          & \\textbf{MAP}             \\\\ \midrule \n"
    for i in range(x, x + 9):
        # print(df.iloc[i,y])
        table += "\\begin{tabular}[c]{@{}l@{}}" + str(df.iloc[i, y]) + " \end{tabular} " + " &   " + \
                 format(df.iloc[i, y + 1]) + "  & " + format(df.iloc[i, y + 2]) + " &   " + \
                 format(df.iloc[i, y + 1+12]) + "  & " + format(df.iloc[i, y + 2+12]) + " &   " + \
                 format(df.iloc[i, y + 1+24]) + "  & " + format(df.iloc[i, y + 2+24]) + " \\\\ \midrule \n"

    table += "\end{tabular} \n"
    table += "\caption{CPM-BERT (our base model) results on PMD dataset compared to conventional (TF-IDF) and current state-of-the-art self-supervised IR models } \n \label{2} \n \end{table} \n"
    return table



def get_latex_exploration_3col(x, y, df,full=False):
    table = "\\begin{table}[h] \centering \\begin{tabular}{@{}ll|ll|ll|ll@{}} \\toprule \n"
    table += "   &    & k=1   &    &  k=10   &   &    k=avg   &  \\\\ \\midrule"
    table += "\\textbf{Positive }    & \\textbf{Negative }  & \\textbf{R-Prec}   & \\textbf{MAP} & \\textbf{R-Prec}   & \\textbf{MAP} & \\textbf{R-Prec}   & \\textbf{MAP}   \\\\ \midrule \n"
    maximum = 10
    if full:
        maximum = 21
    for i in range(x + 1, x + maximum):
        table +=   str(df.iloc[i, y + 1]) + "   & \\ " + \
                 str(df.iloc[i, y + 2]) + "   &  " +\
                  format( df.iloc[i, y + 3]) + "   &   " + format(df.iloc[i, y + 4])+ "   &   " + \
                  format( df.iloc[i, y + 3+12]) + "   &   " + format(df.iloc[i, y + 4+12])+ "   &   "+ \
                  format( df.iloc[i, y + 3+24]) + "   &   " + format(df.iloc[i, y + 4+24]) +  \
                  "    \\\\ \midrule \n"
    table += "\end{tabular} \n"
    table += "\caption{Exploring different techniques for CPM-BERT  and 90\% confidence interval. Positive and negative sampling methods are explained in section \\ref{positive_sampling}}  \n \label{sampling}  \n \end{table} \n"
    return table

def print_tables(path):
    df = pd.read_csv(path)
    print(df.shape)
    print(get_table_3col(3,1,df))
    print(get_latex_exploration_3col(16,1,df))
    # for i in range(1, 26, 6):
    #     k = df.iloc[0,i]
    #     # print(k)
    #     k = str(k)
    #     k = k[2:]
    #     print(get_latex_table2(3, i, df,k))
    #     print()
    #     print(get_latex_table3(16, i, df,k))





