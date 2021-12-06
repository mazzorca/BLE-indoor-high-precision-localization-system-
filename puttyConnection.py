"""
Script python for connecting ot the raspberry and start the python script
only for windows
"""
from subprocess import Popen
import time
import open_putty_and_send_payload as op

if __name__ == '__main__':
    cd1 = 'cd Desktop \n '
    cd2 = 'cd BLE-Beacon-Scanner \n'
    lscommand = 'ls \n'
    poweroff = 'sudo iwconfig wlan0 power off \n'
    # namefile = 'test2605'
    namefile = 'dati0806run0r'
    ipclient = ["192.168.1.2", "192.168.1.29", "192.168.1.48", "192.168.1.6", "192.168.1.26"]

    pld = [cd1, cd2, poweroff]
    print(pld)
    for i in range(len(ipclient)):
        # realtime='python3 SaveData.py '+namefile+str(i+1)+' \n'ssh
        realtime = 'python3 vi .py ' + namefile + str(i + 1) + ' \n'
        pld = [cd1, cd2, poweroff, realtime]
        puttywin = op.OpenPuttyAndSendPayload(pld=pld, title='Putty test', user='pi', pwd='raspberry', host=ipclient[i],
                                              pos=[
                                                  0, 0, 600, 400])
        time.sleep(2)
