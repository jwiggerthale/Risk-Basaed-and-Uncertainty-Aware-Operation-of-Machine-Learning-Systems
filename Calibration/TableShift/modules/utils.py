import datetime


def log(message: str, 
        file: str = 'log_file.txt', 
        ):
    now = datetime.datetime.now()
    with open(file, 'a') as out_file:
        out_file.write(f'{now}\t{message}')
        out_file.write('\n')

