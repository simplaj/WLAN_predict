# WLAN_predict 
## scores
### 命名方法：score后跟评分方式表示单种方法评分结果 scores后跟评分方法表示整合后的所有评分结果 outputs_new 为降维结果 score_new为降维分簇评分结果
## pre
### csv文件为预测结果进行评分得到的分数 png文件为各种预测方式预测结果与真实值的对比
## Scoring method
### clu&scoring
#### 降维分簇手动评分
### function
#### combine ：将各种方法打分数据整合到一个文件
#### ErrorShow ：画出预测结果与真实结果的对比
#### five_show : 画出五种评分方式的结果
#### LoadStruct: 加载数据的类
#### LstmStruct：整合成S2S的Lstm类
#### normalization：将原始数据归一化
## errors
### 文件格式为出现异常行的对比 每两行一组
## check :检查打分是否符合预期
