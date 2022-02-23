a=[1,2,3,4,5]
index = [0,2]
for i in sorted(index,reverse=True):
    a.pop(i)
print([i+2 for i in a])