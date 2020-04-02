def dict_to_list_gen(d):
    for k, v in zip(d.keys(), d.values()):
        if v == None:
            continue
        yield k
        yield v

def dict_to_list(d):
    return list(dict_to_list_gen(d))

def dict_to_str(dictionary):
    string = ''
    for key, value in dictionary.items():
        string += f"{key}: {value}\n"
    return string
