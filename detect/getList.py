import os
def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            print i,item,item[1]
            path = item[1].replace("\\","/")
            line = '%d\t' % item[0]
            line += '%d\t' % item[2]
            line += '%s\n' % path
            fout.write(line)
def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 3:
                print('lst should at least has three parts, but only has %s parts for %s' %(line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception, e:
                print('Parsing lst met error for %s, detail: %s' %(line, e))
                continue

def list_image(root, recursive):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                if(i%1000==0):
                    print i
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath):
                    f1 = os.path.split(fpath)
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1