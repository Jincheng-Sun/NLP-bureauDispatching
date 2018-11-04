import pandas as pd

file1=pd.DataFrame(pd.read_csv("/Users/sunjincheng/Documents/nlpdata/精炼版.csv",encoding='gb18030'),columns=['工号', '处置单位', '地点', '单位'])
print(file1.shape)
file2=pd.read_csv("/Users/sunjincheng/Documents/nlpdata/valid_data_all.csv",encoding='gb18030')
print(file2.shape)
file2=pd.DataFrame(file2[['工单编号','行业分类1级','行业分类2级','行业分类3级','行业分类4级','诉求内容']],columns=['工单编号','行业分类1级','行业分类2级','行业分类3级','行业分类4级','诉求内容'])
print(file2.shape)
file3=pd.merge(file2,file1,on='工单编号')
print(file3.shape)
file2.to_csv("/Users/sunjincheng/Documents/nlpdata/hangyefenji.csv",encoding='gb18030')