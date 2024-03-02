import os


def create_dir(project_name):
    os.makedirs(project_name, exist_ok=True)


def create_dir_log(project_name):
    
    os.makedirs(project_name, exist_ok=True)

    if not os.path.exists(f'{project_name}_log.txt'):
        my_log = open(f'{project_name}_log.txt', 'a')
        my_log.close()
        

def log_stats(results, project_name):
    file = open(f'{project_name}_log.txt', 'a')
    file.write(results + "\n")
    file.close()

