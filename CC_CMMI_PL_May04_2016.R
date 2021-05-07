
  #*************************************************************************************************
  # Statistical Analysis Implementation for CMMI HM Poland
  # Author: Vladimir Savin, Ph.D., Vladimir_Savin@epam.com, EPAM Systems, http://www.epam.com
  # May 04 2016, Version 1.1.1 for Win 7 64-bit;
  # Control Charts (X-Bar-R), Multiple Regression Analysis (RA) and ANOVA are implemented.
  #*************************************************************************************************

##library("graphics", lib.loc="D:/Program Files/R/R-3.0.2/library")
#library("utils", lib.loc="D:/Program Files/R/R-3.0.2/library")
#library("stats", lib.loc="D:/Program Files/R/R-3.0.2/library")
#library("methods", lib.loc="D:/Program Files/R/R-3.0.2/library")
#library("DBI", lib.loc="D:/Program Files/R/R-3.0.2/library")
#library("datasets", lib.loc="D:/Program Files/R/R-3.0.2/library")
#library("grDevices", lib.loc="D:/Program Files/R/R-3.0.2/library")
##
library("qcc", lib.loc="D:/Program Files/R/R-3.2.3/library")

#Input Excel
#library(readxl)
#read_excel("<path to file")
# Verify the package is installed.
#any(grepl("xlsx",installed.packages()))
# Load the library into R workspace.
#library("xlsx")
# Read the first worksheet in the file input.xlsx.
#data <- read.xlsx("D:/data.xlsx", sheetIndex = 1)
#print(data)

#library(xlsx)
#workbook <- "D:/My3.xlsx"
#mydataframe <- read.xlsx(workbook, 1)
#mydataframe

#InputData
#("D:/InputData.csv")
data <- read.table("D:/InputData.csv", header=TRUE, sep=",") 
data
str(data)
#attach(data)
names(data)



#attach(data) 
plot(age, weight)
abline(lm(age~weight))
title("Regression of Age on Weight")
#detach(data)


##RA
 age <- c(1,3,5,2,11,9,3,9,12,3)
 weight <- c(4.4,5.3,7.2,5.2,8.5,7.3,6.0,10.4,10.2,6.1)
 

 
 mean(weight)
 sd(weight)
 cor(age,weight)
 plot(age,weight)
 fit=lm(age~weight) #fit the linear model
 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! fit <- lm(Y ~ ., data = D)
 
 summary(fit)
 plot (fit)

# x <- seq(0.01,3,0.01)
# plot(x,df(x,1,10),type="l",ylim=c(0,1),ylab="f(x)")
# lines(x,df(x,2,10),lty=6,col="red")
# lines(x,df(x,5,10),lty=2,col="green")
# lines(x,df(x,30,10),lty=3,col="blue")
# legend(2,0.9,c("1","2","5","30"),col=(1:4),lty=c(1,6,2,3), title="numerator d.f.")

 predict(fit)

##QCC
 qcc(age, type="xbar.one")
 qcc(weight, type="xbar.one")
