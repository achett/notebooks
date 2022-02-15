# environment setting
import os
from zipfile import ZipFile

if __name__ == "__main__":
    # override here to specify where the data locates, the root directory and the data directory
    root_dir = r'C:\Users\A4023862\OneDrive - Astellas Pharma Inc\data'
    #root_dir = os.path.abspath(os.path.join(__file__, "../../.."))
    data_dir = os.path.join(root_dir, 'pyhealth')
    os.chdir(data_dir)
    print(root_dir, data_dir)

    with ZipFile(os.path.join(data_dir, 'cms.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting cms files now...')
        zip.extractall()
        print('Done!')

    with ZipFile(os.path.join(data_dir, 'mimic.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting mimic demo files now...')
        zip.extractall()
        print('Done!')

    with ZipFile(os.path.join(data_dir, 'image.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting image files now...')
        zip.extractall()
        print('Done!')

    with ZipFile(os.path.join(data_dir, 'ecg.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting ecg files now...')
        zip.extractall()
        print('Done!')

    with ZipFile(os.path.join(data_dir, 'text.zip'), 'r') as zip:
        # # printing all the contents of the zip file
        # zip.printdir()

        # extracting all the files
        print('Extracting text files now...')
        zip.extractall()
        print('Done!')