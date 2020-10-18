import re
from urllib import parse

def convert(str):
    return '![formula](https://latex.codecogs.com/gif.latex?' + parse.quote(str) + ')'

def inconvert(str):
    return str.replace('$', '@@')

def outconvert(str):
    return str.replace('@@', '$')

def rewrite(src, dst):
    with open (src, 'r') as f:
        content = f.read()
        content = re.sub('\$\$(.*?)\$\$', lambda x: convert(x.group(1)), content, flags=re.DOTALL)
        content = re.sub('(```.*?```)', lambda x: inconvert(x.group(1)), content, flags=re.DOTALL)
        content = re.sub('\$\s*(.*?)\s*\$', lambda x: convert(x.group(1)), content)
        content = re.sub('(```.*?```)', lambda x: outconvert(x.group(1)), content, flags=re.DOTALL)

        fo = open(dst, 'w')
        fo.write(content)
        fo.close()

rewrite('./test.md', './test.md')
