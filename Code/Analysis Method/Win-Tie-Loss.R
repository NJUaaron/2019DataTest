### calculate Win/Tie/Loss result base on wilcox and cliff
###   x win y    iff   wil(x,y) < 0.05, cliff(x,y) > 0.147
###   x loss y   iff   wil(x,y) < 0.05, cliff(x,y) < -0.147
###   x tie y    other conditions

library("stats")
library("effsize")

x <- c()
y <- c()

wilcox.test(x, y, paired=TRUE)$p.value
cliff.delta(x, y)$estimate

