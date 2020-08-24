from path import Path
import os

root = Path('.')
f_root = open(root / 'README.md', 'w')
f_root.write('## Introduction' + os.linesep + os.linesep)
f_root.close()
f_summary = open(root / 'SUMMARY.md', 'w')
f_summary.write('# Introduction' + os.linesep + os.linesep + '* [Introduction](README.md)' + os.linesep)
f_summary.close()
f_root = open(root / 'README.md', 'a')
f_summary = open(root / 'SUMMARY.md', 'a')
for dir in root.dirs():
    if dir.stem != '_book' and dir.stem != '.git':
        f = open(dir / 'README.md', 'w')
        f.write('## ' + dir.stem.replace('-', ' ').upper() + os.linesep + os.linesep)
        f.close()
        f_summary.write(os.linesep + '# ' + dir.stem.replace('-', ' ').upper() + os.linesep + os.linesep)
        f_root.write('* [' + dir.stem.replace('-', ' ').upper() + ']' + '(' + dir.stem + '/README.md)' + os.linesep)
        f_summary.write('* [' + dir.stem.replace('-', ' ').upper() + ']' + '(' + dir.stem + '/README.md)' + os.linesep)
        for file in dir.files():
            print(file)
            if file.stem != 'img' and file.stem != 'README':
                f = open(dir / 'README.md', 'a')
                f.write('* [' + file.stem.replace('-', ' ') + '](' + file.basename() + ')' + os.linesep)
                f_summary.write('    * [' + file.stem.replace('-', ' ') + '](' + dir.stem + '/' + file.basename() + ')' + os.linesep)

f_root.close()
f_summary.close()
