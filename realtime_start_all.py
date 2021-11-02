import multiprocessing
import time
import paramiko as paramiko

from realtimeRouterRegressAll import realtime_process
from multiprocessing import Process, Value

readers = ["192.168.1.2",
           "192.168.1.29",
           "192.168.1.48",
           "192.168.1.6",
           "192.168.1.26"]
username = "pi"
password = "raspberry"
s = "python3 BeaconScanner.py"

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # k = paramiko.RSAKey.from_private_key_file(keyfilename)
    # # OR k = paramiko.DSSKey.from_private_key_file(keyfilename)
    #
    # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect(hostname=host, username=user, pkey=k)

    ready = Value('d', False)
    raltime_process = Process(target=realtime_process, args=("asProcess", ready, False))
    raltime_process.start()

    while True:
        time.sleep(1)
        if ready.value:
            break

    readers_connections = []
    for reader in readers:
        print("reader")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(reader, username=username, password=password)
        command = "sudo iwconfig wlan0 power off; cd Desktop/BLE-Beacon-Scanner; python3 BeaconScanner.py"
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)

    raltime_process.join()
