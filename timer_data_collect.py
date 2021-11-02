import os
import time

how_much_position = 9
time_position = 20
change_time = 6
strsadas = "scp r5.tar.gz alessandromadonna@192.168.1.19:r5.tar.gz"

if __name__ == '__main__':
    input()
    time.sleep(2)

    for position in range(how_much_position):
        os.system(f'say "Posizione {position + 1}"')
        time.sleep(time_position)
        os.system(f'say "Cambia Posizione"')
        time.sleep(change_time)
