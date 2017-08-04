import pandas as pd
import numpy as np
import zipfile
import glob

data = [np.zeros(37)] #prints the data for each class and picture

columns = ['Class1.1,', 'Class1.2,','Class1.3,','Class2.1,','Class2.2,','Class3.1,','Class3.2,','Class4.1,','Class4.2,',
           'Class5.1,','Class5.2,','Class5.3,','Class5.4,','Class6.1,','Class6.2,','Class7.1,','Class7.2,','Class7.3,','Class8.1,',
           'Class8.2,','Class8.3,','Class8.4,','Class8.5,','Class8.6,','Class8.7,','Class9.1,','Class9.2,','Class9.3,','Class10.1,',
           'Class10.2,','Class10.3,','Class11.1,','Class11.2,','Class11.3,','Class11.4,','Class11.5,','Class11.6']
#prints column names

index = ['100018','100037','100042','100052','100056'] #print Galaxy Id

df = pd.DataFrame(data, index=index, columns=columns) #format of chart

df.index.name='GalaxyID,' #prints name for the index

df.reset_index(inplace=True)

df.to_csv #csv format

#should print all ids of pictures
# imagelist2 = []
# zip_ref2 = zipfile.ZipFile('images_test_rev1.zip', 'r')
# zip_ref2.extractall("Downloads")
# zip_ref2.close()
# counter2 = 0
# index = imagelist2
# for file2 in glob.glob("Downloads\images_test_rev1\*.jpg"):
#     index.append(file2)