from bs4 import BeautifulSoup
import json
import urllib.request
import sys
#url = 'https://www.libspn.org/tutorials/tutorial_6_mnist_discrete_gd.html'
#response = urllib.request.urlopen(url)
#  for local html file
none, filename = sys.argv
print(filename)
response = open(filename)
name, none = filename.split('.')
print(name)
text = response.read()

soup = BeautifulSoup(text, 'lxml')
# see some of the html
print(soup.div)
dictionary = {'nbformat': 4, 'nbformat_minor': 1, 'cells': [], 'metadata': {}}
for d in soup.findAll("div"):
    if 'class' in d.attrs.keys():
        for clas in d.attrs["class"]:
            if clas in ["text_cell_render", "input_area"]:
                # code cell
                if clas == "input_area":
                    cell = {}
                    cell['metadata'] = {}
                    cell['outputs'] = []
                    cell['source'] = [d.get_text()]
                    cell['execution_count'] = None
                    cell['cell_type'] = 'code'
                    dictionary['cells'].append(cell)

                else:
                    cell = {}
                    cell['metadata'] = {}

                    cell['source'] = [d.decode_contents()]
                    cell['cell_type'] = 'markdown'
                    dictionary['cells'].append(cell)
open(name+'.ipynb', 'w').write(json.dumps(dictionary))
