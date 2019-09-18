import os

rf = open('wav2htk_concat_all.scp', "r")
wf = open('wav2htk_simple.scp', "w")

for line in rf:
    parts = line.split('  ')
    simple_line = parts[0] + '  ' + parts[0][:-3] + 'htk\n'
    wf.write(simple_line)

rf.close()
wf.close()


