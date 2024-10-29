import os
import glob
import re
import tempfile
import subprocess
import shutil
import numpy as np

class MG5Card:

    _sub_pattern = re.compile(r'\$SUB:([a-zA-Z0-9_]*)')
    _width_pattern = re.compile(r'=== Results Summary for [^\n]* ===\s+'
                               r'Width\s*:\s*([0-9.Ee+-]+)\s*\+\-\s*([0-9.Ee+-]+)\s*(\S+)\s+'
                               r'Nb of events\s*:\s*(\d+)')
    _xs_pattern = re.compile(r'=== Results Summary for [^\n]* ===\s+'
                               r'Cross-section\s*:\s*([0-9.Ee+-]+)\s*\+\-\s*([0-9.Ee+-]+)\s*(\S+)\s+'
                               r'Nb of events\s*:\s*(\d+)')

    def __init__(self, path):
        self.path = path
        with open(self.path) as file:
            self.content = file.read()
        self.vars = self._find_vars()

    def _find_vars(self):
        return sorted(set(self._sub_pattern.findall(self.content)))

    def sub(self, vals):
        content = self.content
        for var in self.vars:
            content = content.replace(f'$SUB:{var}', str(vals[var]))
        return content

    def run(self, vals, tmp_workdir=False, capture_output=False):
        if tmp_workdir: vals['workdir'] = tempfile.TemporaryDirectory().name
        content = self.sub(vals)
        datpath = tempfile.mktemp()
        with open(datpath, 'w') as datfile: datfile.write(content)
        proc = subprocess.run(['mg5_aMC', datpath], capture_output=capture_output)
        if tmp_workdir: shutil.rmtree(vals['workdir'])
        os.remove(datpath)
        return proc

    def run_width(self, vals):
        proc = self.run(vals, tmp_workdir=True, capture_output=True)
        output = proc.stdout.decode()
        try:
            width, width_unc, width_unit, nevent = self._width_pattern.search(output).groups()
        except Exception:
            width, width_unc, width_unit, nevent = np.nan, np.nan, 'GeV', 0
            error = proc.stderr.decode()
            print('ERROR: run_width failed', output, error, sep='\n' + '-' * 40 + '\n', flush=True)
        return float(width), float(width_unc), width_unit, int(nevent)

    def run_xs(self, vals):
        proc = self.run(vals, tmp_workdir=True, capture_output=True)
        output = proc.stdout.decode()
        try:
            xs, xs_unc, xs_unit, nevent = self._xs_pattern.search(output).groups()
        except Exception:
            xs, xs_unc, xs_unit, nevent = np.nan, np.nan, 'pb', 0
            error = proc.stderr.decode()
            print('ERROR: run_xs failed', output, error, sep='\n' + '-' * 40 + '\n', flush=True)
        return float(xs), float(xs_unc), xs_unit, int(nevent)

def load_cards(basedir='cards'):
    cards = { }
    for path in glob.glob(os.path.join(basedir, '*.dat')):
        cards[os.path.basename(path)] = MG5Card(path)
    return cards

if __name__ == '__main__':
    for path, card in load_cards().items():
        print(path, card.vars, card.content, sep='\n' + '-' * 40 + '\n', flush=True)
