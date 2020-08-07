def myprint(l, f):
    print(l)
    f.writelines(l + '\n')
    f.flush()