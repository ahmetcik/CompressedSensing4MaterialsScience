#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "21/10/16"

""" Note: the _calc_rdf function was written by Fawzi Roberto Mohamed
(mohamed@fhi-berlin.mpg.de)"""

import ase
import ase.io
import json
import os
import sys
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as pypl
#import condor
from PIL import Image
from sklearn import preprocessing
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer




class NOMADStructure(object):
    """Read the atomic structure and the total energy of a system given a NOMAD
        JSON file.

        Gets data from JSON file and sets up the ase.Atoms objects.
        It handles multiple geometries in the NOMAD file (e.g. configurations
        from a geometry optimization). 

        Parameters
        ----------

        in_file : string,
            Path to json file to read

        frame_list : int, list (or list of lists)
            Specifies for each NOMAD json file the frames to load.
            It is a list if only one frame for each json file needs to be
            loaded, while it is a list of lists otherwise.
            Negative indeces are supported.

        png_path : string
            Path to the folder where the png files for each structure are
            written.

        geo_path : string
            Path to the folder where the png files for each structure are
            written.

        png_file : string
            Path to the png file where the image for each structure and frame
            is written.

        geo_file : string
            Path to the geometry file where the image for each structure and
            frame is written.

        xray_path : string
            Path to the folder where the xray diffraction pattern png files for
            each structure are written.

        xray_file : string
            Path to the xray diffraction pattern png files where the image for
            each structure and frame is written.

        isPeriodic : bool
            Specify if the system is periodic (used for generating periodic
            replicas in the plot).
            If `True` if create a 4x4x4 periodic replica of the unit cell for
            visualization.
            .. todo:: It should be guessed automatically using
            `nomad_classify `.

        file_format : text, optional, {'NOMAD', 'rdf'}
            Specify what is the format of the file to read.
            ('rdf' is DEPRECATED).


        Attributes
        ----------

        atoms : dict

        energy_eV : dict

        name : (?)
            Sha - checksum.

        chemical_symbols : dict

        chemical_formula : dict

        spacegroup_analyzer : dict
            Spacegroup analyzer object of pymatgen
            http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html


        """

    def __init__(
            self, in_file=None, frame_list=None, png_path=None,
            png_file=None, geo_path=None, geo_file=None, desc_path=None,
            rdf_file=None, xray_path=None, xray_file=None, isPeriodic=None,
            file_format=None, cell_type=None, take_first=True,
            descriptor=None):

        self.m_to_ang = 1.0e10
        self.j_to_ev = 6.241509e18
        self.atoms = {}
        self.scaled_positions = {}
        self.energy_total = {}
        self.energy_total__eV = {}
        self.in_file = in_file
        self.frame_list = frame_list
        self.png_path = png_path
        self.png_file = {}
        self.geo_path = geo_path
        self.geo_file = {}
        self.desc_path = desc_path
        self.xray_path = xray_path
        self.xray_file = {}
        self.xray_npy_file = {}
        self.xray_rs_file = {}
        self.rdf_file = {}
        self.name = os.path.splitext(os.path.basename(in_file))[0]
        self.chemical_symbols = {}
        self.chemical_formula = {}
        self.isPeriodic = isPeriodic
        self.Id = {}
        self.isEnergyNone = False
        self.file_format = file_format
        self.cell_type = cell_type
        self.spacegroup_analyzer = {}
        
        
        if self.file_format is None:
            self.file_format = 'NOMAD'

        if self.file_format == 'NOMAD':
            # read JSON file
            json_dict = self._get_json_dict(in_file)

            # get single_config_calc (target: energy_total)
            single_config_calc = self._get_single_config_calc(json_dict)

            # extract energy from the list in single_config_calc
            for (gIndexRun, gIndexSingle), structure in single_config_calc.items():
                energy = structure.get('energy_total')
                if energy is None:
            #        logger.warning(
             #           "Could not find energy_total in " +
              #          "single_config_calculations \n with gIndexSingle {0}" +
               #         "in section_run with gIndexRun {1} \n" +
                #        "in file {2}".format(
                 #           gIndexSingle,
                  #          gIndexRun,
                   #         self.in_file))
                    isEnergyNone = True
                    self.energy_total[gIndexRun, gIndexSingle] = 0.0
                    self.energy_total__eV[gIndexRun, gIndexSingle] = 0.0
                else:
                    self.energy_total[
                        gIndexRun, gIndexSingle] = energy
                    self.energy_total__eV[
                        gIndexRun, gIndexSingle] = energy * self.j_to_ev

            # get system_description (target: structure)
            system_descriptions = self._get_system_descriptions(json_dict)

            # compare dictionaries of single_config_calc and
            # system_descriptions
            #logger.debug("Single configuration keys: {0}".format(
             #   set(single_config_calc.keys())))
            #logger.debug("System description keys: {0}".format(
            #    set(system_descriptions.keys())))

            intersect = set(single_config_calc.keys()).intersection(
                set(system_descriptions.keys()))
            not_intersect = set(single_config_calc.keys()).symmetric_difference(
                set(system_descriptions.keys()))

            #logger.debug(
            #    "Deleting frames that do not both have 'single_config_calc' and in 'system_descriptions'.")
            for key in not_intersect:
                if key in system_descriptions:
                    del system_descriptions[key]
            if take_first:
                #logger.debug(
                #    "Keeping only 1st frame in which both 'single_config_calc' and 'system_descriptions' are present.")
                not_intersect_system = set([(0, 0)]).symmetric_difference(
                    set(system_descriptions.keys()))
                not_intersect_single = set([(0, 0)]).symmetric_difference(
                    set(single_config_calc.keys()))
                index = 0
            else:
                logger.debug(
                    "Keeping only last frame in which both 'single_config_calc' and 'system_descriptions' are present.")
                index = len(system_descriptions)-1
            system_descriptions_value = system_descriptions[(0, index)]
            system_descriptions.clear()
            system_descriptions[(0, 0)] = system_descriptions_value

            single_config_calc_value = single_config_calc[(0, index)]
            single_config_calc.clear()
            single_config_calc[(0, 0)] = single_config_calc_value
               
            # print len(single_config_calc)
            # print len(system_descriptions)

            # print set(single_config_calc.keys())
            # print set(system_descriptions.keys())

            # extract structure data from the list in system_descriptions
            # the key is a tuple (gIndexRun, gIndexDesc)
            # Note: self.atoms is a dict with key (gIndexRun, gIndexDesc)
            # for (gIndexRun, gIndexDesc), structure in
            # system_descriptions.items():
            for (gIndexRun, gIndexDesc), structure in system_descriptions.items():
                # old format
                # labels = structure.get('atom_label')
                labels = structure.get('atom_labels')

                if labels is None:
                    raise Exception(
                        "Could not find atom_label in section_system_description" +
                        "\nwith gIndex %d in section_run with gIndex " +
                        "%d!" %
                        gIndexDesc %
                        gIndexRun)

                # positions = structure.get('atom_position')
                positions = structure.get('atom_positions')

                if positions is None:
                    raise Exception(
                        "Could not find atom_position in " +
                        "section_system_description \nwith gIndex %d in" +
                        "section_run with gIndex %d!" % gIndexDesc % gIndexRun)

                cell = structure.get('simulation_cell')

                # convert to Angstrom and numpy array
                positions = self.m_to_ang * np.asarray(positions)

                # check if all labels are known by ase
                # if not change unknown label to X
                for i in range(len(labels)):
                    number = ase.data.atomic_numbers.get(labels[i])
                    if number is None:
                        labels[i] = u'X'

                if cell is None:
                    self.isPeriodic = False
                    atoms = ase.Atoms(
                        symbols=labels, positions=positions,
                        cell=None, pbc=False)
                else:
                    self.isPeriodic = True
                    # convert to Angstrom and numpy array
                    cell = self.m_to_ang * np.asarray(cell)
                    # in the metadata, simulation_cell has lattice vectors as
                    # columns, ase has them as rows => transpose
                    atoms = ase.Atoms(
                        symbols=labels, positions=positions,
#                        cell=np.transpose(cell), pbc=True)
                        cell=cell, pbc=True)


                # get conventional or primitive standard cell with pymatgen
                if cell is not None:
                    if self.cell_type is not None:
                        if self.cell_type == 'primitive':
                            atoms = get_conventional_standard_atoms(atoms)
                            atoms = get_primitive_standard_atoms(atoms)
                        elif self.cell_type == 'standard':
                            atoms = get_conventional_standard_atoms(atoms)

                # get the SpacegroupAnalyzer object from pymatgen 
                # for further processing (e.g. to get spacegroup number or symbol)
                # (from PyMatGen): tolerance of 0.1 (the value used in Materials Project) is often needed
                spacegroup_analyzer = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms),
#                    symprec=1e-1, angle_tolerance=5) 
                    symprec=1e-3, angle_tolerance=1) 
                
                # ---------------- START EXPERIMENTATION ---------------------

                # ONLY TO EXPERIMENT THINGS
                # let us rotate the cell randomly and see if the get the same pattern
                #atoms.rotate('x', np.pi/6, center=(0, 0, 0))
                #logger.debug('Rotating atoms.')


                '''
                
                xrd_plot, xrd_data = get_xrd_diffraction(atoms)
                
                path = '/home/ziletti/nomad-lab-base/analysis-tools/structural-similarity/tutorials/tmp/'
                
                filename_xrd_1d = os.path.abspath(
                    os.path.normpath(
                        os.path.join(
                            path, '%s_%d_%d_%s' %
                                (self.name, gIndexRun, gIndexDesc, '1d_s_AgKb1.png'))))
                
                #print 'saving file: ', filename_xrd_1d
                axes = xrd_plot.gca()
                axes.set_xlim([0, 90])
                xrd_plot.savefig(filename_xrd_1d)
                pypl.close()
                '''
                # ---------------------END EXPERIMENTATION --------------------
                
                # wrap atomic positions inside unit cell
                atoms.wrap()
                # save the atoms
                self.Id[gIndexRun, gIndexDesc] = (gIndexRun, gIndexDesc)
                self.atoms[gIndexRun, gIndexDesc] = atoms
                # get chemical formula
                self.chemical_formula[
                    gIndexRun,
                    gIndexDesc] = atoms.get_chemical_formula(
                    mode='hill')
                self.chemical_symbols[
                    gIndexRun, gIndexDesc] = atoms.get_chemical_symbols()
                self.scaled_positions[
                    gIndexRun,
                    gIndexDesc] = atoms.get_scaled_positions(
                    wrap=True)
                self.spacegroup_analyzer[
                    gIndexRun,
                    gIndexDesc] = spacegroup_analyzer
                

        elif self.file_format == 'xyz':
            ''' reads xyz file with a list of configurations'''
            xyz_file = open(self.in_file, 'r')

            try:
                ase_frames = ase.io.read(self.in_file, index=':', format='xyz')
                logger.debug(
                    "Number of frames found: {0}".format(
                        len(ase_frames)))

                N_frames = len(ase_frames)

                # usually the frame list is not defined in this cases
                if frame_list is None:
                    frame_list = range(N_frames)

                # check if frame_list is actually a list and not only a number
                if isinstance(frame_list, list) is False:
                    frame_list = [frame_list]

                for frame in frame_list:

                    atoms = ase_frames[frame]

                    # wrap atomic positions inside unit cell
                    atoms.wrap()

                    # save the atoms
                    self.atoms[0, frame] = atoms

                    # instead of chemical formula, display the name
                    self.chemical_formula[
                        0, frame] = atoms.get_chemical_formula(
                        mode='hill')
                    self.energy_total__eV[0, frame] = 0.0
                    self.energy_total[0, frame] = 0.0


                    # it is the same as self.gIndexDescMax = i
                    self.gIndexRunMax = 0
                    self.gIndexDescMax = len(frame_list) - 1

            finally:
                xyz_file.close()

        else:
            logger.error(
                "Please specify if the file is a JSON file is either a file \nfrom the NOMAD repository or a Radial Distribution Function")

    def __getitem__(self, index):
        return self.energy_total[index], self.png_file[index], self.geo_file[index], self.chemical_formula[index], self.name

    def __len__(self):
        return self._data_len

    def _get_json_dict(self, in_file):
        """Reads JSON content from file.
        Args:
            in_file: file which is read.
        Returns:
            Dictionary loaded from JSON file.
        """
        try:
            with open(in_file) as json_file:
                try:
                    return json.load(json_file)
                except Exception:
                    logger.error(
                        "Could not read content from JSON file '%s'! See below for error message." %
                        in_file)
                    raise
                finally:
                    json_file.close()
        except IOError:
            raise Exception("Could not open file '%s'!" % in_file)

    def _get_system_descriptions(self, json_dict):
        """Extract section_system_description from JSON dictionary (to obtain the structure in this case)
        Args:
            json_dict: JSON dictionary containing parsed data.
        Returns:
            Dictionaries of section_system_description for each section_run as dictionary.
            The keys are a tuple (gIndexRun, gIndexDesc) of the different section_run and section_system_description, respectively.
        """

        # old format
        # for section in sections

        found_section_run = False
        sections = json_dict.get('sections')
        section_system_descriptions = {}

        # read section_run and for each section_run read section_system_description
        # gIndexRun:  gIndex for section_run
        # gIndexDesc: gIndex for section_system_description
        if sections is not None:
            for section in sections.values():
                if section.get('name') == 'section_run':
                    found_section_run = True
                    section_run_sections = section.get('sections')
                    gIndexRun = section.get('gIndex')
                    if section_run_sections is not None and gIndexRun is not None:
                        # extract all occurrences of section_system_description
                        for section_run_section in section_run_sections.values():
                            if section_run_section.get(
                                    'name') == 'section_system':
                                # if section_run_section.get('name') ==
                                # 'section_system_description':
                                found_section_system_description = True
                                gIndexDesc = section_run_section.get('gIndex')
                                section_system_descriptions[
                                    gIndexRun, gIndexDesc] = section_run_section
                    else:
                        raise Exception("Could not find key 'sections' or 'gIndex' in section_run!")

            if not found_section_run:
                raise Exception("Could not find any section_run!")
        else:
            raise Exception("Could not find key 'sections' in JSON file!")

        self.gIndexRunMax = gIndexRun
        self.gIndexDescMax = gIndexDesc
        return section_system_descriptions

    def _get_single_config_calc(self, json_dict):
        """Extract _get_single_config_calc from JSON dictionary (to obtain the energy in this case).
        It is the analogue of ' _get_system_descriptions', but for the single_config_calc section.
        Args:
            json_dict: JSON dictionary containing parsed data.
        Returns:
            Dictionaries of _get_single_config_calc for each section_run as dictionary.
            The keys are a tuple (gIndexRun, gIndexSingle) of the different section_run and section_single_configuration_calculation, respectively.
        """

        # old format
        # for section in sections

        found_section_run = False
        sections = json_dict.get('sections')
        section_single_config_calc = {}

        # read section_run and for each section_run read section_system_description
        # gIndexRun:  gIndex for section_run
        # gIndexDesc: gIndex for single_config_calc

        if sections is not None:
            for section in sections.values():
                if section.get('name') == 'section_run':
                    found_section_run = True
                    section_run_sections = section.get('sections')
                    gIndexRun = section.get('gIndex')

                    if section_run_sections is not None and gIndexRun is not None:
                        # extract all occurrences of
                        # section_single_configuration_calculation
                        for section_run_section in section_run_sections.values():
                            if section_run_section.get(
                                    'name') == 'section_single_configuration_calculation':
                                found_section_single_configuration_calculation = True
                                gIndexSingle = section_run_section.get(
                                    'gIndex')
                                section_single_config_calc[
                                    gIndexRun, gIndexSingle] = section_run_section
                    else:
                        raise Exception(
                            "Could not find key 'sections' or 'gIndex' in section_run!")
            if not found_section_run:
                raise Exception("Could not find any section_run!")
        else:
            raise Exception("Could not find key 'sections' in JSON file!")
        return section_single_config_calc

    def write_geometry(
            self,
            path=None,
            filename_suffix='_aims.in',
            format='aims',
            operation_number=0):
        """Writes the coordinates of the structure as text file with the writing routine of ASE.

        Parameters
        ----------
        path : string, optional, default `geo_path`
            Path to the folder where the geometry files are written

        filename_suffix : string, default '_aims.in'
            Suffix added after filename

        format : string, optional, default 'aims'
            Format of the file to be written. Must be a valid ASE format
            (see https://wiki.fysik.dtu.dk/ase/ase/io/io.html#module-ase.io)

        """
        # define the default path as geo_path
        if path is None:
            path = self.geo_path

        if self.file_format == 'NOMAD':
            for (gIndexRun, gIndexDesc), atoms in self.atoms.items():
                if atoms is not None:
                    # filename is the normalized absolute path
                    filename = os.path.abspath(
                        os.path.normpath(
                            os.path.join(
                                path, '%s_%d_%d_op%d%s' %
                                (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))
                                
                    atoms.write(filename, format=format)

                    # store the normalized absolute path to the geometry file
                    # in the class
                    if configs["isBeaker"] == "True":
#                    if False:
                        # only for Beaker Notebook
                        filename = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    '/user/tmp/', '%s_%d_%d_op%d%s' %
                                    (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))
                    else:
                        # to run locally
                        filename = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    path, '%s_%d_%d_op%d%s' %
                                    (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))

                    self.geo_file[gIndexRun, gIndexDesc] = filename


        else:
            raise Exception(
                "Please specify a valid file format. Possible file formats are 'NOMAD' or 'rdf'.")

    def write_png(
            self,
            path=None,
            filename_suffix='.png',
            replicas=None,
            rotation=None,
            operation_number=0):
                
        """Write png images for the structures read from the JSON files. Builds a 4x4x4 supercell
        if the structure is periodic.


        Parameters
        ----------
        path : string, optional, default `png_path`
            Path to the folder where the geometry files are written

        filename_suffix : string, default '.png'
            Suffix added after filename

        replicas: list of 3 integers, default [4,4,4]
            Number of replicas in each direction. Used only if `isPeriodic` is `True`.

        """
        # define the default path as geo_path
        if path is None:
            path = self.png_path

        if rotation:
            rot = '10z,-80x'
        else:
            rot = '0x, 0y, 0z'

        filename_list = []

        if self.file_format == 'NOMAD':
            for (gIndexRun, gIndexDesc), atoms in self.atoms.items():
                if atoms is not None:
                    # filename is the normalized absolute path
                    filename = os.path.abspath(
                        os.path.normpath(
                            os.path.join(
                                path, '%s_%d_%d_op%d_geometry_thumbnail%s' %
                                (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))

                    # if it is periodic, replicate according to the vector
                    # replicas
                    if self.isPeriodic:
                        # set the default to 4 replicas in each direction
                        if replicas is None:
                            replicas = (4, 4, 4)
                    else:
                        replicas = (1, 1, 1)

                    atoms = atoms * replicas

                    # View used to start ag, and find desired viewing angle
                    # rot = '35x,63y,36z'  # found using ag: 'view -> rotate'

                    # Common kwargs for eps, png, pov
                    kwargs = {
                        # text string with rotation (default='' )
                        'rotation': rot,
                        'radii': .50,  # float, or a list with one float per atom
                        'colors': None,  # List: one (r, g, b) tuple per atom
                        'show_unit_cell': 0,   # 0, 1, or 2 to not show, show, and show all of cell
                        'scale': 100,
                    }

                    atoms.write(filename, format='png', **kwargs)

                    if configs["isBeaker"] == "True":
                        # only for Beaker Notebook
                        filename = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    '/user/tmp/', '%s_%d_%d_op%d_geometry_thumbnail%s' %
                                (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))
                    else:
                        # to run locally
                        filename = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    path, '%s_%d_%d_op%d_geometry_thumbnail%s' %
                                    (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))

                    # store the normalized absolute path to the png file in the
                    # class
                    self.png_file[gIndexRun, gIndexDesc] = filename

                    filename_list.append(filename)

                else:
                    logger.error("Could not find atoms in %s" % self.name)

        else:
            logger.error(
                "Please specify a valid file format. Possible formats are 'NOMAD' or 'rdf'.")
        
        return filename_list


    def _calc_rdf(self, gIndexRun, gIndexDesc, maxR, is_prdf=None):
        """Returns the radial distribution function of a single (user specified) frame given a NOMADstructure.
        This function was written by Fawzi Mohamed.
        Args:
            gIndexRun: int. run index of the frame (usually 0)
            gIndexDesc: int. system_description index of the frame
            maxR: float. cut-off radius up to which the atom distances are considered
            is_prdf: bool. If true calculates partial radial distribution function.
                If false calculates radial distribution function (all atom types are the same)
        Returns:
            radD: radial distribution function for a single frame with gIndexRun=gIndexRun and gIndexDesc=gIndexDesc
        Note:
            cell vectors v1,v2,v3 with values in the columns: [[v1x,v2x,v3x],[v1y,v2y,v3x],[v1z,v2z,v3z]]
        """
        if is_prdf is None:
            is_prdf = True

        if self.isPeriodic:
            # in the metadata, the simulation_cell has lattice vectors as columns, ASE has them as rows
            # in the NOMADstructure the cell has been trasposed from the metadata => lattice vectors are as rows
            # calc_rdf needs lattice vectors as colulmns => transpose again
            cell = self.atoms[gIndexRun, gIndexDesc].get_cell()
#            cell = np.transpose(cell)
            atoms = self.atoms[gIndexRun, gIndexDesc]
            positions = self.atoms[
                gIndexRun,
                gIndexDesc].get_positions(
                wrap=True)
            maxR2 = maxR * maxR

            radD = {}
            for ii, a1 in enumerate(atoms):
                for jj, a2 in enumerate(atoms[ii:]):

                    # write atomic numbers if prdf, otherwise set them to zero
                    if is_prdf:
                        n1 = a1.number
                        n2 = a2.number
                    else:
                        n1 = 0
                        n2 = 0

                    label = "%d_%d" % (min(n1, n2), max(n1, n2))
                    if label not in radD:
                        radD[label] = {
                            "particle_atom_numer_1": n1,
                            "particle_atom_numer_2": n2,
                            "arr": []
                        }
                    arr = radD[label]["arr"]
                    r0 = np.array([a1.x - a2.x, a1.y - a2.y, a1.z - a2.z])
                    r02 = np.dot(r0, r0)
                    r = r02 - maxR2
                    m = np.dot(cell.transpose(), cell)
                    l = np.dot(r0, cell)
                    r = np.dot(r0, r0) - maxR2
                    mii = m[0, 0]
                    mij = m[0, 1]
                    mik = m[0, 2]
                    mjj = m[1, 1]
                    mjk = m[1, 2]
                    mkk = m[2, 2]
                    li = l[0]
                    lj = l[1]
                    lk = l[2]
                    c = (
                        mjj**2 *
                        mkk**3 *
                        r -
                        2 *
                        mjj *
                        mjk**2 *
                        mkk**2 *
                        r +
                        mjk**4 *
                        mkk *
                        r -
                        lj**2 *
                        mjj *
                        mkk**3 +
                        lj**2 *
                        mjk**2 *
                        mkk**2 +
                        2 *
                        lj *
                        lk *
                        mjj *
                        mjk *
                        mkk**2 -
                        lk**2 *
                        mjj**2 *
                        mkk**2 -
                        2 *
                        lj *
                        lk *
                        mjk**3 *
                        mkk +
                        lk**2 *
                        mjj *
                        mjk**2 *
                        mkk)
                    a = (
                        mii *
                        mjj**2 *
                        mkk**3 -
                        mij**2 *
                        mjj *
                        mkk**3 -
                        2 *
                        mii *
                        mjj *
                        mjk**2 *
                        mkk**2 +
                        mij**2 *
                        mjk**2 *
                        mkk**2 +
                        2 *
                        mij *
                        mik *
                        mjj *
                        mjk *
                        mkk**2 -
                        mik**2 *
                        mjj**2 *
                        mkk**2 +
                        mii *
                        mjk**4 *
                        mkk -
                        2 *
                        mij *
                        mik *
                        mjk**3 *
                        mkk +
                        mik**2 *
                        mjj *
                        mjk**2 *
                        mkk)
                    b = (
                        2 *
                        li *
                        mjj**2 *
                        mkk**3 -
                        2 *
                        lj *
                        mij *
                        mjj *
                        mkk**3 -
                        4 *
                        li *
                        mjj *
                        mjk**2 *
                        mkk**2 +
                        2 *
                        lj *
                        mij *
                        mjk**2 *
                        mkk**2 +
                        2 *
                        lj *
                        mik *
                        mjj *
                        mjk *
                        mkk**2 +
                        2 *
                        lk *
                        mij *
                        mjj *
                        mjk *
                        mkk**2 -
                        2 *
                        lk *
                        mik *
                        mjj**2 *
                        mkk**2 +
                        2 *
                        li *
                        mjk**4 *
                        mkk -
                        2 *
                        lj *
                        mik *
                        mjk**3 *
                        mkk -
                        2 *
                        lk *
                        mij *
                        mjk**3 *
                        mkk +
                        2 *
                        lk *
                        mik *
                        mjj *
                        mjk**2 *
                        mkk)

                    delta = b * b - 4 * a * c

                    if (a == 0 or delta < 0):
                        continue
                    sDelta = math.sqrt(delta)
                    imin = int(math.ceil((-b - sDelta) / (2 * a)))
                    imax = int(math.floor((-b + sDelta) / (2 * a)))
                    for i in range(imin, imax + 1):
                        cj = (mkk * r + i**2 * mii * mkk + 2 * i * li *
                              mkk - i**2 * mik**2 - 2 * i * lk * mik - lk**2)
                        aj = (mjj * mkk - mjk**2)
                        bj = (2 * i * mij * mkk + 2 * lj * mkk -
                              2 * i * mik * mjk - 2 * lk * mjk)
                        deltaj = bj * bj - 4 * aj * cj
                        if (aj == 0 or deltaj < 0):
                            continue
                        sDeltaj = math.sqrt(deltaj)
                        jmin = int(math.ceil((-bj - sDeltaj) / (2 * aj)))
                        jmax = int(math.floor((-bj + sDeltaj) / (2 * aj)))
                        for j in range(jmin, jmax + 1):
                            ck = r + j**2 * mjj + 2 * i * j * mij + i**2 * mii + 2 * j * lj + 2 * i * li
                            ak = mkk
                            bk = (2 * j * mjk + 2 * i * mik + 2 * lk)
                            deltak = bk * bk - 4 * ak * ck
                            if (ak == 0 or deltak < 0):
                                continue
                            sDeltak = math.sqrt(deltak)
                            kmin = int(math.ceil((-bk - sDeltak) / (2 * ak)))
                            kmax = int(math.floor((-bk + sDeltak) / (2 * ak)))
                            for k in range(kmin, kmax + 1):
                                if (jj != 0 or i != 0 or j != 0 or k != 0):
                                    rr = r02 + k**2 * mkk + k * \
                                        (2 * j * mjk + 2 * i * mik + 2 * lk) + j**2 * mjj + 2 * i * j * mij + i**2 * mii + 2 * j * lj + 2 * i * li
                                    arr.append(math.sqrt(rr))
            wFact = 4 * math.pi * len(atoms) / abs(np.linalg.det(cell))
            for k, v in radD.items():
                v["arr"].sort()
                v["weights"] = map(lambda r: 1.0 / (wFact * r * r), v["arr"])
            return radD

    def write_rdf(
            self,
            path=None,
            filename_suffix='.json',
            maxR=None,
            is_prdf=None):
        """Write a JSON file with structure info and radial distribution functions of all the frames in the NOMADstructure

        One json file is generated for each NOMADjson file. If the NOMADjson file contains multiple frames,
        the partial radial distribution functions of different frames are appended.

        Parameters
        ----------
        path : string
            Path to the folder where the rdf file is written

        filename_suffix : string, default '.json'
            Suffix added after filename (the filename is the NoMaD database unique identifier)

        replicas : list of 3 integers, default [4,4,4]
            Number of replicas in each direction. Used only if `isPeriodic` is `True`.

        maxR : float, default 25
            Cut-off radius in Angstrom up to which the atom distances are considered

            .. todo:: Add BIN_COUNT to parameters.

        is_prdf : bool, default `True`
            If `True` calculates partial radial distribution function.
            If `False` calculates radial distribution function (all atom types are the same)


        Returns
        -------

        filename : string
            Absolute path where the file with the partial radial distribution function(s) is written.


        """

        if self.isPeriodic:
            filename = os.path.abspath(
                os.path.normpath(
                    os.path.join(
                        path, '%s%s' %
                        (self.name, filename_suffix))))
            self.rdf_file = filename
            outF = file(filename, 'w')
            outF.write("""
    {
          "data":[""")

            for (gIndexRun, gIndexDesc), atoms in self.atoms.items():
                if atoms is not None:
                    # filename is the normalized absolute path
                    # filename = os.path.abspath(os.path.normpath(os.path.join(path, '%s_%d_%d%s' % (self.name, gIndexRun, gIndexDesc, filename_suffix))))
                    # store the normalized absolute path to the rdf file in the class
                    # self.rdf_file[gIndexRun, gIndexDesc] = filename

                    cell = self.atoms[gIndexRun, gIndexDesc].get_cell()
#                    cell = np.transpose(cell)
                    atoms = self.atoms[gIndexRun, gIndexDesc]
                    positions = self.atoms[
                        gIndexRun, gIndexDesc].get_positions(
                        wrap=True)
                    energy_total__eV = self.energy_total__eV[gIndexRun, gIndexDesc]
                    energy_total = self.energy_total[gIndexRun, gIndexDesc]

                    writeColon = False

                    rdf = self._calc_rdf(
                        gIndexRun=gIndexRun,
                        gIndexDesc=gIndexDesc,
                        maxR=25,
                        is_prdf=is_prdf)
                    res = {
                        "calc_id": 'NaN',
                        "checksum": self.name,
                        "energy_total": energy_total,
                        "energy_total__eV": energy_total__eV,
                        "path": self.rdf_file,
                        "step": gIndexDesc,
                        "final": 'NaN',
                        "struct_id": 'NaN',
                        "cell": cell.tolist(),
                        "particle_atom_number": map(
                            lambda x: x.number,
                            atoms),
                        "particle_position": map(
                            lambda x: [
                                x.x,
                                x.y,
                                x.z],
                            atoms),
                        "radial_distribution_function": rdf}
                    if (writeColon):
                        outF.write(", ")
                    writeColon = True
                    json.dump(res, outF, indent=2)
            outF.write("""
    ] }""")
            outF.flush()

        else:
            raise Exception(
                "File {0} is a non-periodic structure. The radial distribution function is \ncurrently implemented only for periodic systems.".format(self.in_file))

        # only 1 filename because all the frames are in the same file
        return filename

    def write_xray(
            self,
            path=None,
            filename_suffix='_xray.png',
            replicas=None,
            grayscale=True,
            user_param_source=None,
            user_param_detector=None,
            rotation=None,
            angles_d=None):
        """Write xray images for the structures read from the JSON files. Builds a 4x4x4 supercell
        if the structure is periodic.

        Parameters
        ----------
        path : string, optional, default `xray_path`
            Path to the folder where the xray files are written

        filename_suffix : string, default '_xray.png'
            Suffix added after filename

        replicas: list of 3 integers, default [4,4,4]
            Number of replicas in each direction. Used only if `isPeriodic` is `True`.

        Returns
        -------

        filename_list : string or list
            Xray file (or list of xray files) generated from the file. It is a list if multiple
            frames are present.

        """

        filename_list = []

        # define the default path as xray_path
        if path is None:
            path = self.xray_path

        param_source = {
            'wavelength': 1.0E-11,
            'pulse_energy': 1.6022e-14,
            'focus_diameter': 1E-6
        }

        param_detector = {
            'distance': 0.10,
            'pixel_size': 16E-5,
            'nx': 28,
            'ny': 28
        }

        if user_param_source is not None:
            param_source.update(user_param_source)

        if user_param_detector is not None:
            param_detector.update(user_param_detector)

        if not rotation:
            angles_d = [0.0]

        # logger.debug("Source parameters \n {0}".format(param_source))
        # logger.debug("Detector parameters \n {0}".format(param_source))

        if (self.file_format == 'NOMAD') or (self.file_format == 'xyz'):
            for (gIndexRun, gIndexDesc), atoms in self.atoms.items():
                if atoms is not None:

                    # if it is periodic, replicate according to the vector
                    # replicas
                    if self.isPeriodic:
                        # set the default to 4 replicas in each direction
                        if replicas is None:
                            replicas = (4, 4, 4)
                    else:
                        replicas = (1, 1, 1)

                    atoms = atoms * replicas

                    # Source
                    src = condor.Source(**param_source)

                    # Detector
                    det = condor.Detector(**param_detector)

                    # Atoms
                    atomic_numbers = map(lambda x: x.number, atoms)

                    # convert Angstrom to m (CONDOR uses meters)
                    atomic_positions = map(
                        lambda x: [
                            x.x * 1E-10,
                            x.y * 1E-10,
                            x.z * 1E-10],
                        atoms)

                    for angle_d in angles_d:
                        # filename is the normalized absolute path
                        filename_xray = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    path, '%s_%d_%d_%d%s' %
                                    (self.name, gIndexRun, gIndexDesc, angle_d, filename_suffix))))
                        filename_rs = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    path, '%s_%d_%d_%d%s' %
                                    (self.name, gIndexRun, gIndexDesc, angle_d, '_xray_rs.png'))))
                        filename_ph = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    path, '%s_%d_%d_%d%s' %
                                    (self.name, gIndexRun, gIndexDesc, angle_d, '_xray_ph.png'))))
                        filename_npy = os.path.abspath(
                            os.path.normpath(
                                os.path.join(
                                    path, '%s_%d_%d_%d%s' %
                                    (self.name, gIndexRun, gIndexDesc, angle_d, '.npy'))))

                        angle = angle_d / 360. * 2 * np.pi
                        rotation_axis = np.array([0., 0., 1.])/np.sqrt(1.)
                        #rotation_axis = np.array([1., 1., 1.]) / np.sqrt(3.)
                        quaternion = condor.utils.rotation.quat(
                            angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
                        rotation_values = np.array([quaternion])
                        rotation_formalism = "quaternion"
                        rotation_mode = "extrinsic"

                        par = condor.ParticleAtoms(
                            atomic_numbers=atomic_numbers,
                            atomic_positions=atomic_positions,
                            rotation_values=rotation_values,
                            rotation_formalism=rotation_formalism,
                            rotation_mode=rotation_mode)

                        s = "particle_atoms"
                        E = condor.Experiment(src, {s: par}, det)
                        res = E.propagate()

                        real_space = np.fft.fftshift(
                            np.fft.ifftn(
                                np.fft.fftshift(
                                    res["entry_1"]["data_1"]["data_fourier"])))
                        intensity_pattern = res["entry_1"]["data_1"]["data"]
                        fourier_space = res["entry_1"][
                            "data_1"]["data_fourier"]
                        phases = np.angle(fourier_space) % (2 * np.pi)
                        # vmin = np.log10(res["entry_1"]["data_1"]["data"].max()/10000.)

                        # scaler = preprocessing.StandardScaler().fit(intensity_pattern)
                        # intensity_pattern = scaler.transform(intensity_pattern)

                        # intensity_pattern = np.log10(intensity_pattern)

                        # real space image
                        # np.save(filename_npy, rs8)

                        if grayscale:
                            I8 = (((intensity_pattern -
                                    intensity_pattern.min()) /
                                   (intensity_pattern.max() -
                                    intensity_pattern.min())) *
                                  255.0).astype(np.uint8)
                            rs8 = (((real_space -
                                     real_space.min()) /
                                    (real_space.max() -
                                     real_space.min())) *
                                   255.0).astype(np.uint8)
                            ph8 = (((phases - phases.min()) / (phases.max() -
                                   phases.min())) * 255.0).astype(np.uint8)

                            img = Image.fromarray(I8)
                            img.save(filename_xray)

                            img = Image.fromarray(rs8)
                            img.save(filename_rs)

                            img = Image.fromarray(ph8)
                            img.save(filename_ph)

                            np.save(filename_npy, I8)
                            # np.save(filename_npy, ph8)

                        else:
                            pypl.imsave(
                                filename_xray, np.log10(intensity_pattern))
                            np.save(filename_npy, intensity_pattern)
                            # pypl.imsave(filename_xray, np.log10(intensity_pattern), vmin=vmin)
                            pypl.imsave(filename_ph, phases)
                            pypl.imsave(filename_rs, abs(real_space))

#                       the Viewer with xray files will not work
#                        if configs["isBeaker"] == "True":
#                            # only for Beaker Notebook
#                            filename_xray = os.path.abspath(
#                                os.path.normpath(
#                                    os.path.join(
#                                        '/user/tmp/', '%s_%d_%d_%d%s' %
#                                        (self.name, gIndexRun, gIndexDesc, angle_d, filename_suffix))))

                        # store the normalized absolute path to the png file in
                        # the class
                        self.xray_file[gIndexRun, gIndexDesc] = filename_xray
                        self.xray_npy_file[
                            gIndexRun, gIndexDesc] = filename_npy
                        self.xray_rs_file[gIndexRun, gIndexDesc] = filename_rs

                        filename_list.append(filename_xray)
                        filename_list.append(filename_npy)
                        filename_list.append(filename_rs)

                else:
                    raise Exception("Could not find atoms in %s" % self.name)

        else:
            raise Exception(
                "Please specify a valid file format. Possible format is 'NOMAD'.")

        return filename_list

    def write_target_values(self, path=None, filename_suffix='_target.json',
                            target=None, operation_number=0):
        """Write target values. One file for each frame.
        The target works only if one frame is considered. Please check.

        Parameters
        ----------
        path : string, optional, default `xray_path`
            Path to the folder where the geometry files are written

        filename_suffix : string, default '_target.json'
            Suffix added after filename

        """

        filename_list = []

        # define the default path as xray_path
        if path is None:
            path = self.xray_path

        if (self.file_format == 'NOMAD') or (self.file_format == 'xyz'):
            for (gIndexRun, gIndexDesc), atoms in self.atoms.items():
                if atoms is not None:
                    # filename is the normalized absolute path
                    filename = os.path.abspath(
                        os.path.normpath(
                            os.path.join(
                                path, '%s_%d_%d_op%d%s' %
                                (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))


                    outF = file(filename, 'w')

                    outF.write("""
            {
                  "data":[""")

                    chemical_formula = self.chemical_formula[
                        gIndexRun, gIndexDesc]
                    cell = self.atoms[gIndexRun, gIndexDesc].get_cell()
#                    cell = np.transpose(cell)
                    atoms = self.atoms[gIndexRun, gIndexDesc]
                    energy_total__eV = self.energy_total__eV[gIndexRun, gIndexDesc]
                    energy_total = self.energy_total[gIndexRun, gIndexDesc]
                    spacegroup_symbol = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_space_group_symbol()                    
                    spacegroup_number = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_space_group_number()                    
                    pointgroup_symbol = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_point_group_symbol()                    
                    crystal_system = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_crystal_system() 
                    # Get the lattice for the structure, e.g., (triclinic, orthorhombic, cubic, etc.).
                    #This is the same than the crystal system with the exception of the
                    #hexagonal/rhombohedral lattice
                    lattice_type = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_lattice_type() 
                    hall_symbol = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_hall()                    
                    lattice_centering = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_space_group_symbol()[0]                    
                    symmetry_dataset = self.spacegroup_analyzer[gIndexRun, gIndexDesc].get_symmetry_dataset()                    

                    
                    writeColon = False

                    res = {
                        "checksum": self.name,
                        "main_json_file_name": self.in_file,
                        "chemical_formula": chemical_formula,
                        "energy_total__eV": energy_total__eV,
                        "energy_total": energy_total,
                        "gIndexRun": gIndexRun,
                        "gIndexDesc": gIndexDesc,
                        "cell": cell.tolist(),
                        "particle_atom_number": map(
                            lambda x: x.number,
                            atoms),
                        "particle_position": map(
                            lambda x: [
                                x.x,
                                x.y,
                                x.z],
                            atoms),
                        "filename": filename,
                        "target": target,
                        "spacegroup_symbol": spacegroup_symbol,
                        "spacegroup_number": spacegroup_number,
                        "pointgroup_symbol": pointgroup_symbol,
                        "crystal_system": crystal_system,
                        "hall_symbol": hall_symbol,
                        "lattice_type": lattice_type,
                        "lattice_centering": lattice_centering,
                        "Bravais_lattice_cs": crystal_system+lattice_centering,
                        "Bravais_lattice_lt": lattice_type+lattice_centering,
                        #"symmetry_dataset": symmetry_dataset,
                        "operation_number": operation_number,
                    }

                    if (writeColon):
                        outF.write(", ")

                    writeColon = True
                    json.dump(res, outF, indent=2)

                    outF.write("""
            ] }""")
                    outF.flush()

                else:
                    logger.error("Could not find atoms in %s" % self.name)


                filename = os.path.abspath(
                    os.path.normpath(
                        os.path.join(
                            path, '%s_%d_%d_op%d%s' %
                            (self.name, gIndexRun, gIndexDesc, operation_number, filename_suffix))))

                filename_list.append(filename)

        else:
            raise Exception("Please specify a valid file format.")

        return filename_list


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
