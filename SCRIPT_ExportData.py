from library import Misc

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60)

ind = Misc.read_csv('digitize/ind.csv', 20, 40)
dyad = Misc.read_csv('digitize/dyad.csv', 10, 50)

data1.to_csv('export/data_trial1.csv', index=False)
data5.to_csv('export/data_trial5.csv', index=False)
data10.to_csv('export/data_trial10.csv', index=False)
data15.to_csv('export/data_trial15.csv', index=False)
data20.to_csv('export/data_trial20.csv', index=False)
ind.to_csv('export/data_individual.csv', index=False)
dyad.to_csv('export/data_dyad.csv', index=False)