import os

folder_name = "abrera"
os.chdir('./'+ folder_name)

i=0
for f in os.listdir():
    i+=1
    rename = folder_name + "_" + str(i)+".jpg"
    print(rename)
    os.rename(f, rename)

#based on https://www.youtube.com/watch?v=ve2pmm5JqmI