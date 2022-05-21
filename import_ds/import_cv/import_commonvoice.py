from .import_commonvoice_csv import import_commonvoice_csv

def import_commonvoice(path, toolkit):
    if '.csv' in path:
        return import_commonvoice_csv(path, toolkit)