class DataSplitInfo(object):
       
    def __init__(self,rs):
        self.columns_data = []
        self.ids = []
        for i, row in rs.iterrows():
            ss=[]
            for j, column in row.iteritems():
                if j!='id':
                    ss.append(column)
                else:
                    self.ids.append(column)
            self.columns_data.append(ss)