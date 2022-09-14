def make_dict(bunch):
    bunch_dict = {'x':bunch.x.copy(), 'xp':bunch.xp.copy(), 
                  'y':bunch.y.copy(), 'yp':bunch.yp.copy(), 
                  'z':bunch.z.copy(), 'dp':bunch.dp.copy()}
    
    return bunch_dict
