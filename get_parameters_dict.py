def get_parameters_dict(path_readme):
    with open(path_readme,'r') as f:
        fdata = f.read()
    fdata = fdata.split('\n')

    parameters_dict = dict()
    for row in fdata:
        row = row.split()
        if len(row) > 1:
            key = ''
            for i, word in enumerate(row):
                if word.isalpha():
                    key += word
                    key += ' '
                else:
                    key += word[:-1] 
                    try:
                        parameters_dict[key] = float(row[i+1])
                    except:
                        parameters_dict[key] = row[i+1]
                    break
    return parameters_dict
