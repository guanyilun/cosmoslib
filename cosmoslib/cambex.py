"""This script is originally written by Matthew Hasselfield. I find it
very useful and import it into this lib."""

import time, sys, os, glob
import subprocess as sp
import numpy as np

DEFAULTS = {
    'local_temp_dir': '.cambex',
    'camb_bin': '/global/homes/y/yguan/.conda/envs/myenv/bin/camb',
}

class IniLine(object):
    def __init__(self):
        self.key = None
        self.type = 'unknown'

    @classmethod
    def from_value(cls, key, value):
        self = cls()
        self.type = 'param'
        self.key = key
        self.set(value)
        return self

    @classmethod
    def from_line(cls, line):
        self = cls()
        self.parse(line)
        return self

    def parse(self, text):
        self.text = text
        w = text.split()
        if len(w) == 0:
            self.type = 'blank'
        elif w[0][0] == '#':
            self.type = 'comment'
        elif len(w) < 3 or w[1] != '=':
            self.type = 'unknown'
        else:
            self.type = 'param'
            self.key = w[0]
            self.value_raw = ' '.join(w[2:])

    def set(self, value):
        self.value_raw = self.format_value(value)

    def render(self):
        if self.type == 'param':
            return '%s = %s\n' % (self.key, self.value_raw)
        return self.text

    @staticmethod
    def format_value(value):
        if isinstance(value, str) or not hasattr(value, '__iter__'):
            return IniLine.format_value([value])
        return ' '.join([str(x) for x in value])

        
class IniFile(object):
    def __init__(self):
        self.lines = []
        self.lookup = {}

    def read(self, filename):
        for line in open(filename):
            new_l = IniLine.from_line(line)
            if new_l.type == 'param':
                i = self.lookup.get(new_l.key)
                if i is None:
                    i = len(self.lines)
                    self.lines.append(new_l)
                    self.lookup[new_l.key] = i
                else:
                    self.lines[i] = new_l

    def regenerate_lookup(self):
        new_lookup = {}
        for i, line in enumerate(self.lines):
            if line.type == 'param':
                new_lookup[line.key] = i
        self.lookup = new_lookup

    def set(self, key, value, warn_new=True):
        i = self.lookup.get(key)
        if i is not None:
            self.lines[i].set(value)
        else:
            if warn_new:
                print('Parameter %s not previously known' % key)
            self.lookup[key] = len(self.lines)
            self.lines.append(IniLine.from_value(key, value))

    def get(self, key):
        return self.lines[self.lookup[key]].value_raw

    def set_new(self, key, value):
        return self.set(key, value, warn_new=False)

    def write(self, filename=None):
        if filename is None:
            fout = sys.stdout
        else:
            fout = open(filename, 'w')
        for line in self.lines:
            fout.write(line.render())

class CambResult(object):
    def __init__(self, stuff):
        values = []
        for k,v in stuff:
            setattr(self, k, v)
            values.append(v)
        self.values = np.vstack(values)

class CambSession:
    def __init__(self, base_ini=None, rank=None,
                 camb_bin=None, output_dir=None):
        if rank is None:
            lead = 'X'
        else:
            lead = str(rank)
        if output_dir is None:
            output_dir = DEFAULTS['local_temp_dir']
        self.output_dir = output_dir
        if camb_bin is None:
            camb_bin = DEFAULTS['camb_bin']
        self.camb_bin = camb_bin
        self.output_prefix = lead

    def get_output_filename(self, base=None):
        out = os.path.join(self.output_dir, self.output_prefix)
        if base is None:
            return out
        return out + '_' + base

    def set_base_params(self, source):
        if isinstance(source, IniFile):
            pass
        elif isinstance(source, str):
            x_ = IniFile()
            x_.read(source)
            source = x_
        elif isinstance(source, dict):
            source = IniFile()
            source.update(source)
        else:
            raise ValueError("source is what?")
        self.ini = source

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # Write params
        self.ini.set('output_root', os.path.abspath(self.get_output_filename()))
        paramf = os.path.abspath(self.get_output_filename('params_in.ini'))
        self.ini.write(paramf)
        # Run CAMB
        camb_bin = self.camb_bin
        camb_bind, camb_bine = os.path.split(camb_bin)
        p = sp.Popen([camb_bin, paramf], cwd=camb_bind,
                     stdout=sp.PIPE, stderr=sp.PIPE)
        X = p.communicate()
        return CambResult([
                ('ok', p.returncode==True),
                ('returncode', p.returncode),
                ('stdout', X[0]),
                ('stderr', X[1]),
                ('output_dir', self.output_dir)])

    def set_transfer_redshifts(self, z=None, enable=True):
        if z is None:
            self.redshifts = None
            self.ini.set('get_transfer', 'F')
            return
        self.redshifts = np.array(z)
        assert (self.redshifts.ndim in [0,1])
        self.many_redshifts = (self.redshifts.ndim > 0)
        self.redshifts.shape = -1 # promote to 1d even if scalar.
        self.redshift_order = self.redshifts.argsort()[::-1]
        # Remove existing transfer keys.
        to_kill = []
        for k,v in self.ini.lookup.items():
            if k.startswith('transfer_redshift(') or k.startswith('transfer_filename('):
                to_kill.append(v)
        for v in sorted(to_kill, reverse=True):
            self.ini.lines.pop(v)
        self.ini.regenerate_lookup()
        del to_kill
        # Add in the ones we want
        for i, j in enumerate(self.redshift_order):
            self.ini.set_new('transfer_redshift(%d)'%(i+1), self.redshifts[j])
            self.ini.set_new('transfer_filename(%d)'%(i+1), 'transfer%d' % (i+1))
            self.ini.set_new('transfer_matterpower(%d)'%(i+1), 'matterpower%d' % (i+1))
        self.ini.set('transfer_num_redshifts', len(self.redshifts))
        self.ini.set('get_transfer', 'T')

    def load_transfers(self):
        if self.redshifts is None:
            raise RuntimeError("Transfer redshifts not set prior to run.")
        k, P = None, None
        for i, j in enumerate(self.redshift_order):
            mp = self.get_output_filename('matterpower%d' % (i+1))
            k_, P_ = np.loadtxt(mp, unpack=1)
            if k is None:
                k = k_
                P = np.empty((len(self.redshifts), len(k)))
            else:
                assert np.all(k==k_)
            P[j] = P_
        k2, T = None, None
        for i, j in enumerate(self.redshift_order):
            mp = self.get_output_filename('transfer%d' % (i+1))
            kT_ = np.loadtxt(mp, unpack=1)
            k_, T_ = kT_[0], kT_[1:]
            if k2 is None:
                k2 = k_
                T = np.empty((len(self.redshifts),) + T_.shape)
            else:
                assert np.all(k2==k_)
            T[j] = T_
        return (CambResult([
                    ('k', k), ('P', P)]),
                CambResult([
                    ('k', k2), ('T', T)]))

    def load_scalarCls(self):
        mp = self.get_output_filename('scalCls.dat')
        data = np.loadtxt(mp, unpack=1)
        stuff = [('ell', data[0]),
                 ('TT', data[1]),
                 ('EE', data[2]),
                 ('TE', data[3])]
        return CambResult(stuff)

    def load_vectorCls(self):
        mp = self.get_output_filename('vecCls.dat')
        data = np.loadtxt(mp, unpack=1)
        stuff = [('ell', data[0]),
                 ('TT', data[1]),
                 ('EE', data[2]),
                 ('BB', data[3]),
                 ('TE', data[4])]
        
        return CambResult(stuff)    
    
    def load_tensorCls(self):
        mp = self.get_output_filename('tensCls.dat')
        data = np.loadtxt(mp, unpack=1)
        stuff = [('ell', data[0]),
                 ('TT', data[1]),
                 ('EE', data[2]),
                 ('BB', data[3]),
                 ('TE', data[4])]
        
        return CambResult(stuff)    

    def cleanup(self):
        files = glob.glob(self.get_output_filename('*'))
        for f in files:
            os.remove(f)
        try:
            os.rmdir(self.output_dir)
        except OSError:
            pass
