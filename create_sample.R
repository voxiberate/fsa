options(scipen = 999)
######################

df_size=1000

cs_df <- data.frame(
  
  age=c(sample(c("18-24", "25-34", "35-44", "45-54", "55-64", "65+"), df_size, replace=TRUE, prob=c(0.05,0.35,0.1,0.2,0.2,0.05))),
  salary = c(rnorm(df_size, mean=50000, sd=10000)),
  debt_to_income_ratio= (c(rnorm(df_size, mean=0.3, sd=0.1))),
  gender=c(sample( LETTERS[c(6,13)], df_size, replace=TRUE, prob=c(0.45, 0.55) )),
  Years_of_employment=c(runif(df_size, min=0, max=30)),
  employment_status=c(sample(c("military", "private", "government", "self employed", "retired", "unemployed"), df_size, replace=TRUE, prob=c(0.05,0.35,0.1,0.2,0.2,0.05))),
  Nationality=c(sample(c("Afroamerican","Hispanic","White"), df_size, replace=TRUE, prob=c(0.1,0.4,0.5))),
  Residency_status=c(sample(c("local", "sponsored", "visa"), df_size, replace=TRUE, prob=c(0.35,0.2,0.45))),
  Marital_Status= c(sample(c("married", "single", "divorced"), df_size, replace=TRUE, prob=c(0.40,0.55,0.05))),
  Educational_Level= c(sample(c("Doctoral or professional degree","Masters degree","Bachelor's degree","Postsecondary nondegree award","High school diploma or equivalent","No formal educational credential"),df_size,replace=TRUE,prob=c(0.05,0.20,0.525,0.15,0.05,0.025))),
  Outstanding_Debt = c(rnorm(df_size, mean=120000, sd=50000)),
  Sector_Employed =c(sample(c("Energy", "Materials", "Industrials","Consumer Discretionary","Consumer Staples","Health Care","Financials","Information Technology","Telecommunication","Services","Utilities","Real Estate"),df_size, replace =TRUE)),
  Politically_Exposed = c(sample(c("Yes", "No"), df_size, replace=TRUE, prob=c(0.01,0.99))),
  Region = c(sample(c("Miami Beach", "Miami Center", "Suburbs", "County", "Other"), df_size, replace=TRUE, prob=c(0.20,0.05,0.10,0.05,0.50)))
)

#Debt to Income cannot be below 0 
cs_df$debt_to_income_ratio[cs_df$debt_to_income_ratio < 0] = 0


#fix variable type:
cs_df$age<-as.factor(cs_df$age)
cs_df$gender<-as.factor(cs_df$gender)
cs_df$employment_status<-as.factor(cs_df$employment_status)
cs_df$Nationality<-as.factor(cs_df$Nationality)
cs_df$Residency_status<-as.factor(cs_df$Residency_status)
cs_df$Marital_Status<-as.factor(cs_df$Marital_Status)
cs_df$Educational_Level<-as.factor(cs_df$Educational_Level)
cs_df$Politically_Exposed   <- as.factor(cs_df$Politically_Exposed)
cs_df$Residence <- as.factor(cs_df$Region)
cs_df$Sector_Employed <- as.factor(cs_df$Sector_Employed)


#check the column stats and type:
names(cs_df)
str(cs_df)
summary(cs_df)

#just have a look at some variables:
dev.off()
par(mfrow=c(4,3))

hist(cs_df$salary, main="salary")
hist(cs_df$debt_to_income_ratio, main="debt_to_income_ratio")
hist(cs_df$Outstanding_Debt, main="Outstanding_Debt")

barplot(prop.table(table(cs_df$age)), main='age group')
barplot(prop.table(table(cs_df$gender)), main='gender')
barplot(prop.table(table(cs_df$Nationality)), las=2,  main='Nationality')
barplot(prop.table(table(cs_df$employment_status)), las=2, main='employment_status')

barplot(prop.table(table(cs_df$Residence)), las=2, main='Residence')
barplot(prop.table(table(cs_df$Educational_Level)), las=2, main='Educational_Level')
barplot(prop.table(table(cs_df$Politically_Exposed)), main='Politically_Exposed')
barplot(prop.table(table(cs_df$Marital_Status)), main='Marital_Status')


#add unique_id:
cs_df$pool_member_identifier <- 1:nrow(cs_df)

########
#
write.csv(cs_df, file="stress_test_correspondents.csv", row.names = F)
