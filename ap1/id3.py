import numpy as np

class Compass:
    '''
    Decision compass class used in the ID3 algorithm. Used in building the decision trees.
    '''
    def __init__(self):
        self.leaf = False
        pass
    
    def set_test_treshold(self, value):
        self.test = value

    def set_test_index(self, test_index):
        self.index = test_index

    def set_leaf(self, val):
        self.leaf = True
        self.val = val

    def set_left(self, child):
        self.left = child
        
    def set_right(self, child):
        self.right = child
        

    def search(self, test_instance):
        if self.leaf:
            return self.val

        if test_instance[self.index] < self.test:
            if self.left.leaf:
                return self.left.val
            else:
                return self.left.search(test_instance)
        else:
            if self.right.leaf:
                return self.right.val
            else:
                return self.right.search(test_instance)

class ID3:
    '''
    ID3 algorithm class
    '''
    def __init__(self,data_labels_tuple, split_treshold):
        '''
        Args:
            data_labels_tuple : data + label
            split_treshold : When to make another split in the decision tree. This is some sort of pruning, limiting the depth of the decision tree.
        '''
        self.data = np.array(data_labels_tuple[0],dtype='float')
        self.labels = np.array(data_labels_tuple[1],dtype='float')
        self.split_treshold = split_treshold

        self.root = Compass()
        self.build_tree(self.data,self.labels,self.root)

    def build_tree(self, data :np.array, labels:np.array, c:Compass):

        d = data
        l = labels

        if l.size < self.split_treshold:
            avg = l.sum() / l.shape[0]
            c.set_leaf(avg)
            return

        #we first get the attribute count
        attr_count = d.shape[1]

        i        = -1  #chosen index for node
        j0        = 0  
        min_loss = 0   #minimum sum of mean square errros
        treshold = 0   #treshold where minimum is achieved

        for index in range(attr_count):

            sd = np.sort(d[:,index])
            #we now put a treshold between two adjacent points and we calc the MSE
            for j in range(0,l.shape[0]-1):
                i1 = sd[j];   l1 = l[j]
                i2 = sd[j+1]; l2 = l[j+1]

                if l1 == l2:
                    continue

                t = (i1 + i2) / 2

                left = l[:(j+1)].flatten()
                right = l[(j+1):].flatten()

                avg_left  = left.sum()   / left.size
                avg_right = right.sum() / right.size

                left  -= avg_left
                right -= avg_right
                err = np.dot(left,left) + np.dot(right,right)
                
                if min_loss > err or i == -1:    
                    i = index
                    j0 = j+1
                    min_loss = err
                    treshold = t                            


        #print(i,j0, min_loss, treshold)

        c.set_test_treshold(treshold)
        c.set_test_index(i)

        lc = Compass()
        rc = Compass()
        c.set_left(lc)
        c.set_right(rc)

        self.build_tree(d[:j0],l[:j0],lc)
        self.build_tree(d[j0:],l[j0:],rc)

    def predict(self, test_instance):
        return self.root.search(test_instance)