###Control Charts for Visual Studio 2015 R version 3.2.3 !

  #*************************************************************************************************
  # Statistical Analysis Implementation for CMMI HM Poland
  # Author: Vladimir Savin, Ph.D., Vladimir_Savin@epam.com, EPAM Systems, http://www.epam.com
  # June 14 2016, Version 1.1.1 for Win 7 64-bit;
  # Control Charts (X-Bar-R), Multiple Regression Analysis(RA), and ANOVA are implemented.
  #*************************************************************************************************

library("qcc", lib.loc="D:/Program Files/R/R-3.2.3/library")

# Input data
data = read.csv("D:/mydata.csv")  #read csv file

Defect <- data$Defects
Defects_Phase2 <- data$Defects_Phase2
Velocity <- data$Velocity

 qcc(Defect, type="xbar.one", std.dev = "SD", ylab = "Defects", xlab = "Sprint")
 qcc(Defects_Phase2, type="xbar.one", std.dev = "SD", ylab = "Defects", xlab = "Sprint")
 qcc(Velocity, type="xbar.one", std.dev = "SD", ylab = "Velocity", xlab = "Sprint" )
 qcc(x, type="xbar.one", std.dev = "SD", ylab = "Velocity", xlab = "Sprint")

################################################
# Multiple Linear Regression Analysis
 fit <- lm(Defect ~ Velocity + Heads, data= data)
 summary(fit) # show results
 plot(fit )
# Other useful functions !!!
 coefficients(fit) # model coefficients

# ANOVA
 anova(fit) # anova table
                                                                                                                                                                           

######################################################################

#ANOVA
anova(lm(Defect ~ Velocity + Heads, data= data))