

def predict_multiletask(i_val,prev_h_shared,c_next_shared, prev_h, c_next,cell,taskname):
    prev_h_shared, c_next_shared,\
        prev_h, c_next, \
            = cell(i_val,(prev_h_shared,c_next_shared),(prev_h,c_next), taskname)
    return prev_h_shared,c_next_shared,\
           prev_h, c_next

def predict_singletask(i_val, prev_h, c_next,cell):

    prev_h, c_next= cell(i_val,(prev_h,c_next))
    return prev_h, c_next