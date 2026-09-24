# import pandas as pd
# df = pd.read_csv('project/data/wine.csv')
# c = []
# n = []
# for x in df.columns:
#     if df[x].dtype == 'object':
#        c.append(x)
#        #print(f'- name: {x}',f'  dtype: {df[x].dtype}',f'  values: {df[x].unique()}',sep='\n')
#     else:
#        # print(f'- name: {x}', f'  dtype: {df[x].dtype}', f'  min: {min(df[x].unique())}',f'  max: {max(df[x].unique())}', sep='\n')
#         n.append(x)
# for x in n:
#     print(f'- name: {x}', f'  dtype: {df[x].dtype}', f'  min: {min(df[x].unique())}',f'  max: {max(df[x].unique())}', sep='\n')
