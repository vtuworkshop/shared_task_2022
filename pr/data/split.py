import glob
import os


def copy(src, dst):
    with open(src) as f:
        data = f.read()
    with open(dst, 'w') as f:
        f.write(data)


def split(first, second):
    first_paths = glob.glob(f'{first}/*.conll')
    second_paths = glob.glob(f'{second}/*.conll')

    first_files = {os.path.basename(x) for x in first_paths}
    second_files = {os.path.basename(x) for x in second_paths}

    shared = first_files.intersection(second_files)

    first_not_shared = sorted(first_files.difference(shared))
    second_not_shared = sorted(second_files.difference(shared))

    for x in shared:
        copy(f'{first}/{x}', f'test/{x}')

    for x in first_not_shared[:10]:
        copy(f'{first}/{x}', f'test/{x}')
    for x in first_not_shared[10:20]:
        copy(f'{first}/{x}', f'dev/{x}')
    for x in first_not_shared[20:]:
        copy(f'{first}/{x}', f'train/{x}')

    for x in second_not_shared[:10]:
        copy(f'{second}/{x}', f'test/{x}')
    for x in second_not_shared[10:20]:
        copy(f'{second}/{x}', f'dev/{x}')
    for x in second_not_shared[20:]:
        copy(f'{second}/{x}', f'train/{x}')


if __name__ == '__main__':
    pairs = [
        ('aid', 'ian'),
        ('ashuman', 'basema'),
        ('sian', 'tam'),
    ]

    for x, y in pairs:
        split(x, y)
