# Part A
#install.packages("tm")
library(tm)

sms_raw <- read.csv("C:\\Users\\shree\\OneDrive\\Documents\\Quarter 6\\Predictive Analytics\\Week 2\\sms_spam.csv", stringsAsFactors = FALSE)
View(sms_raw)
str(sms_raw)

sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)

sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:2], as.character)

sms_corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
removePunctuation("hello...world")

#install.packages("SnowballC")
library("SnowballC")
wordStem(c("learn", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(tolower = TRUE,
  removeNumbers = TRUE,
  stopwords = TRUE,
  removePunctuation = TRUE,
  stemming = TRUE
))

sms_dtm
sms_dtm2


sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#install.packages("wordcloud")
library("wordcloud")
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

findFreqTerms(sms_dtm_train, 5)
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)
sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}


sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,convert_counts)

#install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)
sms_test_pred <- predict(sms_classifier, sms_test)

#install.packages("gmodels")
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))

#Improving model
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels,laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_test_labels,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))






#Part 2
df <- read.csv("C:\\Users\\shree\\OneDrive\\Documents\\Quarter 6\\Predictive Analytics\\Week 2\\unlabelled.csv", stringsAsFactors = FALSE)
View(df)
str(df)
df$label <- factor(df$label)
table(df$label)

library(tm)
df_corpus <- VCorpus(VectorSource(df$title))
print(df_corpus)
inspect(df_corpus[1:2])
as.character(df_corpus[[1]])
lapply(df_corpus[1:2], as.character)

df_corpus_clean <- tm_map(df_corpus,content_transformer(tolower))
as.character(df_corpus[[1]])
as.character(df_corpus_clean[[1]])

df_corpus_clean <- tm_map(df_corpus_clean, removeNumbers)

df_corpus_clean <- tm_map(df_corpus_clean,removeWords, stopwords())

df_corpus_clean <- tm_map(df_corpus_clean, removePunctuation)

library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

df_corpus_clean <- tm_map(df_corpus_clean, stemDocument)
df_corpus_clean <- tm_map(df_corpus_clean, stripWhitespace)

as.character(df_corpus[[1]])
as.character(df_corpus_clean[[1]])


df_dtm <- DocumentTermMatrix(df_corpus_clean)

df_dtm2 <- DocumentTermMatrix(df_corpus, control = list(tolower = TRUE,
                                                          removeNumbers = TRUE,
                                                          stopwords = TRUE,
                                                          removePunctuation = TRUE,
                                                          stemming = TRUE
))

df_dtm
df_dtm2


df_dtm_train <- df_dtm[1:16452, ]
df_dtm_test <- df_dtm[16453:21936, ]

df_train_labels <- df[1:16452, ]$label
df_test_labels <- df[16453:21936, ]$label

prop.table(table(df_train_labels))
prop.table(table(df_test_labels))

#install.packages("wordcloud")
library("wordcloud")
wordcloud(df_corpus_clean, min.freq = 50, random.order = FALSE)

notclickbait <- subset(df, label == "not-clickbait")
clickbait <- subset(df, label == "clickbait")

wordcloud(notclickbait$title, max.words = 40, scale = c(3, 0.5))
wordcloud(clickbait$title, max.words = 40, scale = c(3, 0.5))

findFreqTerms(df_dtm_train, 5)
df_freq_words <- findFreqTerms(df_dtm_train, 5)
str(df_freq_words)
df_dtm_freq_train<- df_dtm_train[ , df_freq_words]
df_dtm_freq_test <- df_dtm_test[ , df_freq_words]
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}


df_train <- apply(df_dtm_freq_train, MARGIN = 2,convert_counts)
df_test <- apply(df_dtm_freq_test, MARGIN = 2,convert_counts)



library(e1071)
df_classifier <- naiveBayes(df_train, df_train_labels)

df_test_pred <- predict(df_classifier, df_test)

library(gmodels)
CrossTable(df_test_pred, df_test_labels,prop.chisq = FALSE, prop.t = FALSE,dnn = c('predicted', 'actual'))


#Improving model
df_classifier2 <- naiveBayes(df_train, df_train_labels,laplace = 1)

df_test_pred2 <- predict(df_classifier2, df_test)

CrossTable(df_test_pred2, df_test_labels,prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,dnn = c('predicted', 'actual'))


