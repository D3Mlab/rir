# Mahdi Abdollahpour, 16/03/2022, 01:55 PM, PyCharm, Neural_PM

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

sns.set_context('paper')

# bert_name = 'VANILLA'
# sim = 'cosine'


name_dict = {'TASB': 'TAS-B',
             'VANILLA': 'BERT',
             'bert#base#uncased': 'BERT',
             'bert-base-uncased': 'BERT',
             '#content#finetune#0$finetune$IR(F)$bert#$SR(F)$LS(F)$HN(F)$SQ(F)$SQS(F)$PC(F)$Mon#Jun#20#09:16:51#2022#LM': 'ConPM, Same Item, In-Batch',
             }


def get_data(key, data):
    d = list([data[key]['NA'], data[key][1], data[key][3], data[key][10], data[key][20], data[key]['avg']])
    return d


def get_data2(data):
    sss = ['NA', 1, 3, 10, 20, 'avg']
    dd = []
    for key in [1, 3, 10, 20, 'avg']:
        for ss in sss:
            sa = ss
            if ss == 'NA':
                sa = 'No Segmentation'
            else:
                sa = '$k_S$ = ' + str(sa)
            dd.append([key, sa, data[key][ss]])
        # d = list([data[key]['NA'],data[key][1],data[key][3],data[key][10],data[key][20],data[key]['avg']])
    return dd


def plot_results(bert_name, sim, df):
    values = df

    bn = name_dict[bert_name]
    data = {k: {kk: 0 for kk in [1, 3, 10, 20, 'avg', 'NA']} for k in [1, 3, 10, 20, 'avg']}
    # print(len(values))
    i = 0
    for val in values.iterrows():

        # print([val[1], val[2], val[5]])
        # map the LM name
        # print(val)
        # print(val[1])
        val = val[1]
        if val['BERT prefernce_matching'] == bert_name and val['similarity'] == sim:
            print(i)
            i += 1
            # print('hay')
            print(val)

            # print(rak,sak)
            # ra = val[4]
            performance = val['R-Prec']
            performance = performance * 100
            print(performance)
            if val['Review aggregation'] == 'topk':
                rak = int(val['k_R'])
                if val['Sentence aggregation'] == 'topk':
                    sak = int(val['k_S'])
                    data[rak][sak] = performance
                elif val['Sentence aggregation'] == 'avg':
                    data[rak]['avg'] = performance
                else:
                    data[rak]['NA'] = performance
            else:
                if val['Sentence aggregation'] == 'topk':
                    sak = int(val['k_S'])
                    data['avg'][sak] = performance
                elif val['Sentence aggregation'] == 'avg':
                    data['avg']['avg'] = performance
                else:
                    data['avg']['NA'] = performance

    print(data)
    a = get_data2(data)
    new_df = pd.DataFrame(a, columns=['Review Aggregation', 'Sentence Aggregation', 'R-Prec'])

    # create plot
    # new_df['Error'] = 0.035
    new_df['Error'] = 0.0
    fig, ax = plt.subplots(figsize=(15, 7))
    g = sns.barplot(x='Review Aggregation', y='R-Prec', hue='Sentence Aggregation', data=new_df,
                    palette='magma',
                    # fill=False,edgecolor='black',
                    # order = ['male', 'female'],
                    capsize=0.05,
                    saturation=8,
                    ax=ax,
                    yerr=[2, 2, 2, 2, 2],
                    # ecolor='y'
                    )
    # title = bn + ', ' + sim.replace('product', 'Product')
    title = bn
    g.axes.set_title(title, fontsize=20, fontweight="bold")
    g.set_xlabel("Review Fusion ($k_R$)", fontsize=20)
    g.set_ylabel('R-Prec', fontsize=30)
    plt.legend(loc='lower right', title='Sentence Fusion', prop={'size': 20}, ncol=3)
    g.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(42, 55)
    plt.savefig('./results/plots/' + bn + '.png')
    plt.show()


def plot_size(path):
    df = pd.read_csv(path)
    df = df.sort_values(by='size', axis=0)
    y = df['R-Prec'].values
    x = df['size'].values
    ci = 0.01
    plt.plot(x, y)
    plt.fill_between(x, (y - ci), (y + ci), color='blue', alpha=0.1)
    plt.show()


def plot_results2(bert, conpm,metric='R-Prec'):
    bert = bert[~bert['Sentence aggregation'].isin(['topk','avg'])]
    bert = bert[bert['similarity']=='dot']

    conpm = conpm[~conpm['Sentence aggregation'].isin(['topk','avg'])]
    conpm = conpm[conpm['similarity']=='dot']
    # print(bert['Sentence aggregation'].head())
    bert = bert.sort_values(by=['k_R'], axis=0)
    conpm = conpm.sort_values(by=['k_R'], axis=0)

    a = bert[metric].values
    b = conpm[metric].values
    kr = bert['k_R'].values
    kr = [str(k) for k in kr]
    kr[-1] = 'avg'
    print(kr)
    data = []
    for i,k in enumerate(kr):
        if k==np.nan:
            k = 'avg'
        if k!='avg':
            k = int(float(k))
        data.append(['BERT',k,a[i]*100])
        data.append(['CPM-BERT, Same Item, In Batch',k,b[i]*100])
    print(data)
    new_df = pd.DataFrame(data, columns=['Review Aggregation', 'LM', metric])
    print(new_df.columns)
    # create plot
    # new_df['Error'] = 0.035
    new_df['Error'] = 0.0
    fig, ax = plt.subplots(figsize=(10, 8))
    g = sns.barplot(x='Review Aggregation', y=metric, hue='LM', data=new_df,
                    palette='magma',
                    # fill=False,edgecolor='black',
                    # order = ['male', 'female'],
                    capsize=0.05,
                    saturation=8,
                    ax=ax,
                    yerr=[0,2],
                    # ecolor='y'
                    )
    # title = bn + ', ' + sim.replace('product', 'Product')
    title = 'Late Fusion'
    g.axes.set_title(title, fontsize=30, fontweight="bold")
    g.set_xlabel("LM", fontsize=30)
    g.set_ylabel(metric, fontsize=30)
    plt.legend(loc='lower right', title='Review Fusion ($k_R$)', prop={'size': 20}, ncol=2)
    g.tick_params(axis='both', which='major', labelsize=20)
    plt.ylim(20, 65)
    # plt.ylim(10, 55)
    plt.savefig('./results/plots/' + 'latefusion_'+metric + '.png')
    plt.show()
