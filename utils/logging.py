# https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()