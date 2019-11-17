def ToInsertLines(did):
    r = []
    bs = ""
    for k in did.keys():
        if bs !="" : bs += ","
        bs+=k
    bs = "INSERT INTO virsta.welldata_norm (" + bs + " ) values"
    n = len(did["id"])
    for i in range(n):
        tmp = ""
        for k in did.keys(): 
             if tmp !="" : tmp += ","
             tmp += str(did[k][i]).replace('[','').replace(']','')
        tmp = bs + " ("+tmp+")"
        r.append(tmp)
    return r
