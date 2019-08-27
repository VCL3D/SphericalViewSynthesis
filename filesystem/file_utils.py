import os

'''
    Filesystem class
    provides file control utilities like tensor saving etc.
'''
class Filesystem:
    def __init__(self):
        self.cwd = os.getcwd()
        if os.path.isfile(self.cwd):
            self.cwd = os.path.basename(self.cwd)
    ''' 
        Creates directory 
        either by giving the absolute path to create
        or the relative path w.r.t. the current working directory

        \param path the path to create
    '''
    def mkdir(self, path):
        if os.path.isabs(path):
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            pathToCreate = os.path.join(self.cwd, path)
            if not os.path.exists(pathToCreate):
                os.mkdir(pathToCreate)