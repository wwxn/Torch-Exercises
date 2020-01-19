import random

path="data.txt"
with open(path,'w') as f:
    for i in range(1,21):
        x=random.random()*10
        y=random.random()*10
        if x<5:
            z=0
        else:
            z=1
        string=str(x)+','+str(y)+','+str(z)+'\n'
        f.write(string)
    f.close()
