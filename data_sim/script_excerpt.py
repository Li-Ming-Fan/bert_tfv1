

#
file_train_bak = "./data_sim/train_sentiment_bak.txt"
file_test_bak = "./data_sim/test_sentiment_bak.txt"

file_train = "./data_sim/train_sentiment.txt"
file_test = "./data_sim/test_sentiment.txt"


file_input = file_train_bak
file_output = file_train
num_output = 3200
#
fp = open(file_input, "r", encoding="utf-8")
lines = fp.readlines()
fp.close()
print(len(lines))
for line in lines[:10]:
    print(line)
#
fp = open(file_output, "w", encoding="utf-8")
for line in lines[:num_output]:
    fp.write("%s" % line)
fp.close()
#


file_input = file_test_bak
file_output = file_test
num_output = 320
#
fp = open(file_input, "r", encoding="utf-8")
lines = fp.readlines()
fp.close()
print(len(lines))
for line in lines[:10]:
    print(line)
#
fp = open(file_output, "w", encoding="utf-8")
for line in lines[:num_output]:
    fp.write("%s" % line)
fp.close()
#



