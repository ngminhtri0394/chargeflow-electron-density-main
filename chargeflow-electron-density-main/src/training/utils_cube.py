import time
import numpy as np
import ase
import io
import zlib
import tarfile
import tempfile
import os
import gzip
# import lz4.frame
from ase.calculators.vasp import VaspChargeDensity
import ase.io.cube
import ase.units

class CubeWriter():
    def __init__(self, filename, atoms, data_shape, origin, comment):
        """
        Function to write a cube file. This is a copy of ase.io.cube.write_cube but supports
        textIO buffer

        filename: str object
            File to which output is written.
        atoms: Atoms object
            Atoms object specifying the atomic configuration.
        data_shape: array-like of dimension 1
            Shape of the data to come
        origin : 3-tuple
            Origin of the volumetric data (units: Angstrom)
        comment : str, optional (default = None)
            Comment for the first line of the cube file.
        """

        self.fileobj = open(filename, "w")
        self.data_shape = data_shape
        self.numbers_written = 0

        if comment is None:
            comment = 'Cube file from ASE, written on ' + time.strftime('%c')
        else:
            comment = comment.strip()
        self.fileobj.write(comment)

        self.fileobj.write('\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')

        if origin is None:
            origin = np.zeros(3)
        else:
            origin = np.asarray(origin) / ase.units.Bohr

        self.fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'
                      .format(len(atoms), *origin))

        for i in range(3):
            n = data_shape[i]
            d = atoms.cell[i] / n / ase.units.Bohr
            self.fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n, *d))

        positions = atoms.positions / ase.units.Bohr
        numbers = atoms.numbers
        for Z, (x, y, z) in zip(numbers, positions):
            self.fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'
                          .format(Z, 0.0, x, y, z))

    def write(self, data):
        for el in data:
            self.numbers_written += 1
            self.fileobj.write("%e\n" % el)

        if self.numbers_written >= np.prod(self.data_shape):
            self.fileobj.close()

def _decompress_tarmember(tar, tarinfo):
    """Extract compressed tar file member and return a bytes object with the content"""

    bytesobj = tar.extractfile(tarinfo).read()
    if tarinfo.name.endswith(".zz"):
        filecontent = zlib.decompress(bytesobj)
    elif tarinfo.name.endswith(".lz4"):
        filecontent = lz4.frame.decompress(bytesobj)
    elif tarinfo.name.endswith(".gz"):
        filecontent = gzip.decompress(bytesobj)
    else:
        filecontent = bytesobj

    return filecontent


def _calculate_grid_pos(density, origin, cell):
    # Calculate grid positions
    ngridpts = np.array(density.shape)  # grid matrix
    grid_pos = np.meshgrid(
        np.arange(ngridpts[0]) / density.shape[0],
        np.arange(ngridpts[1]) / density.shape[1],
        np.arange(ngridpts[2]) / density.shape[2],
        indexing="ij",
    )
    grid_pos = np.stack(grid_pos, 3)
    grid_pos = np.dot(grid_pos, cell)
    grid_pos = grid_pos + origin
    return grid_pos


def _decompress_file(filepath):
    if filepath.endswith(".zz"):
        with open(filepath, "rb") as fp:
            f_bytes = fp.read()
        filecontent = zlib.decompress(f_bytes)
    elif filepath.endswith(".lz4"):
        with lz4.frame.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    elif filepath.endswith(".gz"):
        with gzip.open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    else:
        with open(filepath, mode="rb") as fp:
            filecontent = fp.read()
    return filecontent


def _read_vasp(filecontent):
    # Write to tmp file and read using ASE
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpdeepdft")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vasp_charge = VaspChargeDensity(filename=tmppath)
    os.remove(tmppath)
    density = vasp_charge.chg[-1]  # separate density
    atoms = vasp_charge.atoms[-1]  # separate atom positions

    return density, atoms, np.zeros(3)  # TODO: Can we always assume origin at 0,0,0?

def _read_cube(filecontent):
    textbuf = io.StringIO(filecontent.decode())
    cube = ase.io.cube.read_cube(textbuf)
    # sometimes there is an entry at index 3
    # denoting the number of values for each grid position
    origin = cube["origin"][0:3]
    # by convention the cube electron density is given in electrons/Bohr^3,
    # and ase read_cube does not convert to electrons/Å^3, so we do the conversion here
    cube["data"] *= 1.0 / ase.units.Bohr**3
    return cube["data"], cube["atoms"], origin

def write_cube(fileobj, atoms, data=None, origin=None, comment=None):
    """
    Function to write a cube file. This is a copy of ase.io.cube.write_cube but supports
    textIO buffer

    fileobj: file object
        File to which output is written.
    atoms: Atoms object
        Atoms object specifying the atomic configuration.
    data : 3dim numpy array, optional (default = None)
        Array containing volumetric data as e.g. electronic density
    origin : 3-tuple
        Origin of the volumetric data (units: Angstrom)
    comment : str, optional (default = None)
        Comment for the first line of the cube file.
    """

    if data is None:
        data = np.ones((2, 2, 2))
    data = np.asarray(data)

    if data.dtype == complex:
        data = np.abs(data)

    if comment is None:
        comment = 'Cube file from ASE, written on ' + time.strftime('%c')
    else:
        comment = comment.strip()
    fileobj.write(comment)

    fileobj.write('\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')

    if origin is None:
        origin = np.zeros(3)
    else:
        origin = np.asarray(origin) / ase.units.Bohr

    fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'
                  .format(len(atoms), *origin))

    for i in range(3):
        n = data.shape[i]
        d = atoms.cell[i] / n / ase.units.Bohr
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n, *d))

    positions = atoms.positions / ase.units.Bohr
    numbers = atoms.numbers
    for Z, (x, y, z) in zip(numbers, positions):
        fileobj.write('{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'
                      .format(Z, 0.0, x, y, z))

    for el in data.flat:
        fileobj.write("%e\n" % el)


def write_cube_to_tar(tar, atoms, cubedata, origin, filename):
    """write_cube_to_tar
    Write cube file to tar archive and compress the file using zlib.
    Cubedata is expected to be in electrons/A^3 and is converted to
    electrons/Bohr^3, which is cube file convention

    :param tar:
    :param atoms:
    :param cubedata:
    :param origin:
    :param filename:
    """
    cbuf = io.StringIO()
    write_cube(
        cbuf,
        atoms,
        data=cubedata*(ase.units.Bohr**3),
        origin=origin,
        comment=filename,
    )
    cbuf.seek(0)
    cbytes = cbuf.getvalue().encode()
    # cbytes = zlib.compress(cube_bytes)
    fsize = len(cbytes)
    cbuf = io.BytesIO(cbytes)
    cbuf.seek(0)
    tarinfo = tarfile.TarInfo(name=filename)
    tarinfo.size = fsize
    tar.addfile(tarinfo, cbuf)
