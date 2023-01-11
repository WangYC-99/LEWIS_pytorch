fileName = '/zhangpai25/wyc/lewis/data/noneTyped/train.tsv'

text = ''
lines = []
items = []
results = []

with open(fileName, 'r') as f:
    text = f.read()
lines = text.split('\n')

for l in lines:
    items.append(l.split('\t'))

counter = 0

for i in items:
    if counter == 0:
        counter = counter + 1
        continue
    results.append(i[1])
    if counter == 17000:
        break
    counter = counter + 1

with open('./noneType_large.txt', 'w') as f:
    for each in results:
        f.write(each + '\n')
f.close()