#!/usr/bin/env python3

import glob,sys,re

def get_simfile_prop(basestr):

    simfile = glob.glob(basestr+'*')

    if len(simfile)==0:
        print('No simulation file found!')
        sys.exit()
    elif len(simfile) > 1:
        print('Multiple simulation files found:')
        for k,simf in enumerate(simfile):
            print('[' + str(k+1) + ']  ' + simf)

        filenum = input('Please choose the file to use by its number.')
        try:
            filenum = int(filenum) - 1
        except:
            print('Could not parse number!')
            sys.exit()

        simfile = simfile[filenum]
    else:
        simfile = simfile[0]

    timestamp_regex = re.compile('[\-T:\.0-9]+(?=\.np)')

    timestamp = timestamp_regex.findall(simfile)[0]

    return simfile, timestamp
