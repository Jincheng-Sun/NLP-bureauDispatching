import pandas as pd


def extractdata(dataset, per):
    n = dataset.shape[0]
    testset = pd.DataFrame(dataset.sample(frac=per))
    return testset

#
valid_data = pd.read_csv("/Users/sunjincheng/Documents/nlpdata/new精炼版.csv", encoding="gb18030")

valid_data = pd.DataFrame(valid_data, columns=['工单编号','行业分类1级','行业分类2级','行业分类3级','行业分类4级','诉求内容', '处置单位', '地点', '单位'])

print(valid_data)
testset = extractdata(valid_data, 0.2)

trainset = valid_data.drop(testset.axes[0])

trainset = trainset.sample(frac=1)



testset.to_csv("/Users/sunjincheng/Documents/nlpdata/newtestset.csv",encoding='gb18030')
print(testset.shape)
trainset.to_csv("/Users/sunjincheng/Documents/nlpdata/newtrainset.csv",encoding='gb18030')
print(trainset.shape)

def splitset(testset):
    testset1=extractdata(testset,1/3)
    testset=testset.drop(testset1.axes[0])
    testset2=extractdata(testset,1/3)
    testset3=testset.drop(testset2.axes[0])
    return testset1,testset2,testset3

testdata = pd.read_csv("/Users/sunjincheng/Documents/nlpdata/newtestset.csv",encoding='gb18030')
print(testdata.shape)
testdata = pd.DataFrame(testdata, columns=['工单编号','行业分类1级','行业分类2级','行业分类3级','行业分类4级','诉求内容', '处置单位', '地点', '单位'])

test1,test2,test3=splitset(testdata)
print(test1.shape)
test1.to_csv("/Users/sunjincheng/Documents/nlpdata/testset1.csv",encoding='gb18030')
test2.to_csv("/Users/sunjincheng/Documents/nlpdata/testset2.csv",encoding='gb18030')
test3.to_csv("/Users/sunjincheng/Documents/nlpdata/testset3.csv",encoding='gb18030')