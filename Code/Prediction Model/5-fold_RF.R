#进行5-fold rf, 将预测结果保存到文件
library (randomForest)
set.seed(2019)

rf <- function(filepath, savepath)
{
    label <- "bug"

    flist <- list.files(filepath)   #找出filepath路径下所有的文件
    fnum <- length(flist)

    dir.create(savepath)

    for (f in 1:fnum){  #对每个文件进行操作
        
        #读入数据
        filename <- paste(filepath, flist[f], sep="")
        Data <- read.csv(filename, header=TRUE)
        rows <- nrow(Data)

        #将数据分为5段
        thres <- as.integer(c(1,rows*1/5,rows*2/5,rows*3/5,rows*4/5,rows))

        #5 times 5-fold
        for (i in 1:5){
            train <- Data[-(thres[i]:thres[i+1]),]
            test <- Data[thres[i]:thres[i+1],]

            #将bug列的数据类型从numeric改为factor，否则randomForest函数会视为回归问题而非分类问题
            train[[label]] <- as.factor(train[[label]])
            test[[label]] <- as.factor(test[[label]])

            rf <- randomForest(bug ~ ., data=train, ntree=100, importance=TRUE, proximity=TRUE)

            #在test上进行测试
            pre <- predict(rf, type="prob", newdata = test)[,2]
            output <- data.frame(loc=test$loc, bug=as.integer(test[[label]])-1, pre=pre)

            #输出到文件
            savefile <- paste(savepath, substring(flist[f], 1, nchar(flist[f])-4), "_r" , i-1, ".csv", sep="")
            write.csv(output, file=savefile, row.names=FALSE, quote=FALSE)  
        }
    }
    print("Done!")
}