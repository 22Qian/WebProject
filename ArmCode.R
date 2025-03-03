library(viridis)
library(arules)
library(TSP)
library(data.table)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)
library(RColorBrewer)

# Set working directory and file path
setwd("D:/OM_PhD_CUBoulder/Classes/25S_MachineLearning/M2/ARM")
file_path <- "D:/OM_PhD_CUBoulder/Classes/25S_MachineLearning/M2/ARM/HotelTransactions.csv"

# Read transactions
HotelTrans <- read.transactions(file_path,
                                rm.duplicates = FALSE, 
                                format = "basket", 
                                sep=",", 
                                cols=1)

# Inspect transactions
inspect(HotelTrans[1:5])  # Inspect first 5 transactions

# ---- Apply Apriori Algorithm ----
HrulesK <- apriori(HotelTrans, parameter = list(support = 0.15, 
                                                confidence = 0.15, 
                                                minlen = 2))

# Inspect rules if they exist
if (length(HrulesK) > 0) {
  inspect(HrulesK[1:10])  # View first 10 rules
} else {
  print("No association rules found. Try lowering support and confidence thresholds.")
}

# ---- Frequent Item Analysis ----
# Plot the top 20 most frequent items
itemFrequencyPlot(HotelTrans, topN = 20, type = "absolute")

# Plot relative item frequency
arules::itemFrequencyPlot(HotelTrans, topN = 10,
                          col = brewer.pal(8, 'Pastel2'),
                          main = 'Relative Service Frequency Plot',
                          type = "relative",
                          ylab = "Item Frequency (Relative)")

# ---- Sort Rules by Different Metrics ----
SortedRulesKSup <- sort(HrulesK, by = "support", decreasing = TRUE)
SortedRulesKConf <- sort(HrulesK, by = "confidence", decreasing = TRUE)
SortedRulesKLift <- sort(HrulesK, by = "lift", decreasing = TRUE)

# Inspect sorted rules if they exist
if (length(SortedRulesKSup) > 0) {
  inspect(SortedRulesKSup[1:min(15, length(SortedRulesKSup))])
  inspect(SortedRulesKConf[1:min(15, length(SortedRulesKConf))])
  inspect(SortedRulesKLift[1:min(15, length(SortedRulesKLift))])
} else {
  print("No sorted rules found.")
}

# ---- Summary Check ----
# Ensure `summary` uses an existing object
if (length(SortedRulesKLift) > 0) {
  print(summary(SortedRulesKLift))
} else {
  print("No rules available to summarize.")
}

# ---- Selecting Targeted Rules ----
# Rules where RHS contains "Mountain View"
MountainViewRules <- apriori(data = HotelTrans,
                             parameter = list(supp = 0.001, conf = 0.01, minlen = 2),
                             appearance = list(default = "lhs", rhs = "Mountain View"),
                             control = list(verbose = FALSE))

MountainViewRules <- sort(MountainViewRules, decreasing = TRUE, by = "confidence")

if (length(MountainViewRules) > 0) {
  inspect(MountainViewRules[1:min(4, length(MountainViewRules))])
} else {
  print("No rules found for 'Mountain View'.")
}

# Rules where LHS contains "Room Service"
RoomServiceRules <- apriori(data = HotelTrans,
                            parameter = list(supp = 0.0001, conf = 0.001, minlen = 2),
                            appearance = list(default = "rhs", lhs = "Room Service"),
                            control = list(verbose = FALSE))

RoomServiceRules <- sort(RoomServiceRules, decreasing = TRUE, by = "support")

if (length(RoomServiceRules) > 0) {
  inspect(RoomServiceRules[1:min(4, length(RoomServiceRules))])
} else {
  print("No rules found for 'Room Service'.")
}

# ---- Visualization ----
# Subset the top 100 rules by lift for visualization
HsubrulesK <- head(SortedRulesKLift, 100)

# Check if there are enough rules before plotting
if (length(HsubrulesK) > 0) {
  plot(HsubrulesK)
  plot(HsubrulesK, method = "graph", engine = "interactive")
  plot(HsubrulesK, method = "graph", engine = "htmlwidget")
} else {
  print("Not enough rules to visualize.")
}
