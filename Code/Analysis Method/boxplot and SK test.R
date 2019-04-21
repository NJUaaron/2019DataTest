library("ScottKnott")
library("ggplot2")


#读取数据
data <- read.csv("读取的csv文件")


#计算sk（分类）
sk <-SK(x=data,
        model='AUC~model',
        which='model',
        dispersion = 's')
x1<- summary(sk)


#绘制箱线图
#无配色
qplot(model, AUC, data=data, geom="boxplot")

#手工配色
ggplot(data, aes(x=model, y=AUC, fill=model)) + geom_boxplot() 
        + scale_fill_manual(values=c("red","yellow","grey","purple"))

#特定高亮
data$hl <- "no"     #HightLight
data$hl[data$model=="RF"] <- "yes"
ggplot(data, aes(x=model, y=AUC, fill=hl)) + geom_boxplot() 
        + scale_fill_manual(values=c("grey","#FFDDCC"), guide=FALSE)