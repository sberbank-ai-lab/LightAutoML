#!/usr/bin/env python3

import os
import argparse
import subprocess
from typing import List, Tuple


TShellCmd = str


class LAMAController:
    """Class that provide installation of lama."""
    LAMA_VENV = './lama_venv'
    EXTRA_SECTIONS = ('nlp', 'cv')
    EXTRA_FLAGS = (*('full', 'dev'), *EXTRA_SECTIONS)

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._create_parser()
        self._args = self._parser.parse_args()
        self._validate_extras()
        self._poetry_install_args: str = ""

        if self._debug:
            print('DEBUG: parsed arguments = {}'.format(self._args))

    def execute(self):
        """Engine."""

        print("--- Execution ---")
        for idx, sectio_info in enumerate(self.executable_sections):
            section_name, section_code = sectio_info
            if self._debug:
                section_fmt = "Section #{SI}\nName:\n{SN}\nCode:\n{SC}\n"
                print(section_fmt.format(SI=idx, SN=section_name, SC=section_code))
            else:
                print("Exec: {}".format(section_name))
                subprocess.call(section_code, shell=True, executable='bash')

    def _validate_extras(self):
        for e in self._args.extra:
            if not e in self.EXTRA_FLAGS:
                raise RuntimeError("Wrong extra flag '{}'".format(e))

    @property
    def executable_sections(self) -> List[Tuple[str, TShellCmd]]:
        """List of executable commands.

        Returns:
            cmds: List of commands (section_name, section_code)

        """
        sections = []

        if self._args.all:
            self._args.install = True
            self._args.dist = True
            self._args.docs = True
            self._args.test = True

        if not os.path.exists(self._args.venv):
            sections.append( ('create_venv', self._create_venv()) )
        if self._args.install:
            sections.append( ('install_lib', self._install()) )
        if self._args.dist:
            sections.append( ('build_dist', self._build_dist()) )
        if self._args.docs:
            sections.append( ('build_docs', self._build_docs()) )
        if self._args.test:
            sections.append( ('testing', self._test()))

        return sections

    @property
    def poetry_install_flags(self) -> str:
        install_flags = []

        if self._args.all or "full" in self._args.extra:
            install_flags = ['-E {}'.format(e) for e in self.EXTRA_SECTIONS]
        else:
            need_dev_deps = 'dev' in self._args.extra or self._args.docs
            install_flags = ['-E {}'.format(e) for e in self._args.extra if e != 'dev']
            if not need_dev_deps:
                install_flags.append('--no-dev')

        install_flags = " ".join(install_flags)

        return install_flags

    def _create_parser(self):
        """Create arguments parser."""
        class LAMAHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter
        ):
            pass

        self._parser = argparse.ArgumentParser(
            description='LightAutoML (LAMA)',
            epilog=(
                'Example of use:\n\n'
                '\t./build.py --install    # Install core part of LAMA\n'
                '\t./build.py -p /usr/bin/python3.7 --venv ./.venv --install -e dev -e nlp -e cv'
                '    # Create venv using concrete python, install developer, nlp, cv dependencies\n'
                '\t./build.py --all    # Run full pipeline: create venv, install all dependencies, build dist, build docs\n'
            ),
            formatter_class=LAMAHelpFormatter
        )
        self._parser.add_argument(
            '-p', '--python', default='python3',
            help='Path to python interpretator'
        )
        self._parser.add_argument(
            '--venv', default=self.LAMA_VENV, help='Path to virtual enviroment (Always must be specified if not default)'
        )
        self._parser.add_argument(
            '-a', '--all', action='store_true', help='Make all actions: install, build dist and docs (other option will be ignored)'
        )
        self._parser.add_argument(
            '-i', '--install', action='store_true', help="Install core part of LAMA (excluding: nlp, cv, dev dependencies"
        )
        self._parser.add_argument(
            '-e', '--extra', default=[], action='append', help="Extra dependencies"
        )
        self._parser.add_argument(
            '-b', '--dist', action='store_true', help='Build the source and wheels archives'
        )
        self._parser.add_argument(
            '-d', '--docs', action='store_true', help='Build and check docs'
        )
        self._parser.add_argument(
            '-t', '--test', action='store_true', help='Run tests'
        )

    def _create_venv(self) -> str:
        """Command: create venv and install some dependecies.

        Returns:
            cmd: Shell cmd.

        """
        cmd_fmt = (
            '{PYTHON_EXE} -m venv {VENV}\n'
            'source {VENV}/bin/activate\n'
            'pip install -U pip\n'
            'pip install -U poetry\n'
        )

        cmd = cmd_fmt.format(
            PYTHON_EXE=self._args.python,
            VENV=self._args.venv
        )

        return cmd

    def _install(self) -> str:
        """Command: install the requirements packages.

        Returns:
            cmd: Shell cmd.

        """
        cmd_fmt = (
            'source {VENV}/bin/activate\n'
            'poetry lock\n'
            'poetry install {INSTALL_ARGS}\n'
        )

        cmd = cmd_fmt.format(
            VENV=self._args.venv,
            INSTALL_ARGS=self.poetry_install_flags
        )

        return cmd

    def _build_dist(self) -> str:
        """Command: build dist.

        Returns:
            cmd: Shell cmd.

        """
        cmd_fmt = (
            'source {VENV}/bin/activate\n'
            'rm -rf dist\n'
            'poetry build\n'
        )

        cmd = cmd_fmt.format(
            VENV=self._args.venv,
        )

        return cmd

    def _build_docs(self) -> str:
        """Command: build and check documents.

        Returns:
            cmd: Shell cmd

        """
        cmd_fmt = (
            'source {VENV}/bin/activate\n'
            'cd docs # Build docs\n'
            'mkdir _static\n'
            'make clean html\n'
            'cd ..\n'
            'echo "===== Start check_docs.py ====="\n'
            'python check_docs.py\n'
        )

        cmd = cmd_fmt.format(
            VENV=self._args.venv
        )

        return cmd

    def _test(self) -> str:
        """Command: run tests.

        Returns:
            cmd: Shell cmd

        """
        cmd_fmt= (
            'PACKAGE_NAME=LightAutoML_LAMA\n'
            'source {VENV}/bin/activate\n'
            'cd tests\n'
            'pytest demo*\n'
            'cd ..\n'
        )

        cmd = cmd_fmt.format(
            VENV=self._args.venv
        )

        return cmd


if __name__ == "__main__":
    lama_controller = LAMAController(debug=True)
    lama_controller.execute()
