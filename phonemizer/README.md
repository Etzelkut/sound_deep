[![Travis (.org)](https://img.shields.io/travis/bootphon/phonemizer)](
https://travis-ci.org/bootphon/phonemizer)
[![Codecov](https://img.shields.io/codecov/c/github/bootphon/phonemizer)](
https://codecov.io/gh/bootphon/phonemizer)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/bootphon/phonemizer)](
https://github.com/bootphon/phonemizer/releases/latest)
[![DOI](https://zenodo.org/badge/56728069.svg)](
https://doi.org/10.5281/zenodo.1045825)

# Phonemizer -- *foʊnmaɪzɚ*

* For simplicity use Phonemizer.ipynb

* The phonemizer allows simple phonemization of words and texts in many languages.

* Provides both the `phonemize` command-line tool and the Python function
  `phonemizer.phonemize`.

* It is using four backends: espeak, espeak-mbrola, festival and segments.

  * [espeak-ng](https://github.com/espeak-ng/espeak-ng) supports a lot of
    languages and IPA (International Phonetic Alphabet) output.

  * [espeak-ng-mbrola](https://github.com/espeak-ng/espeak-ng/blob/master/docs/mbrola.md)
    uses the SAMPA phonetic alphabet instead of IPA but does not preserve word
    boundaries.

  * [festival](http://www.cstr.ed.ac.uk/projects/festival) currently supports
    only American English. It uses a [custom
    phoneset](http://www.festvox.org/bsv/c4711.html), but it allows tokenization
    at the syllable level.


## Installation

### Dependencies

* You need to install
  [festival](http://www.festvox.org/docs/manual-2.4.0/festival_6.html#Installation),
  [espeak-ng](https://github.com/espeak-ng/espeak-ng#espeak-ng-text-to-speech)
  and [mbrola](https://github.com/numediart/MBROLA) on your system. On
  Debian/Ubuntu simply run:

        $ sudo apt-get install festival espeak-ng mbrola


### Phonemizer

* The simplest way is using pip:

        $ pip install phonemizer

* **OR** install it from sources with:

        $ git clone https://github.com/bootphon/phonemizer
        $ cd phonemizer
        $ [sudo] python setup.py install

  If you get an error such as `No module named 'segments'
  delete file segments.py from path phonemizer\phonemizer\backend

### Testing

When installed from sources or whithin a Docker image, you can run the tests
suite from the root `phonemizer` folder (once you installed `pytest`):

    $ pip install pytest
    $ pytest


## Python usage

In Python import the `phonemize` function with `from phonemizer import
phonemize`


## Command-line examples

**The above examples can be run from Python using the `phonemize` function**


For a complete list of available options, have a:

    $ phonemize --help

See the installed backends with the `--version` option:

    $ phonemize --version
    phonemizer-2.2
    available backends: espeak-ng-1.49.3, espeak-mbrola, festival-2.5.0, segments-2.0.1


### Input/output exemples

        phonemize("hello world", backend = ='espeak')
        həloʊ wɜːld
	
	phonemize("hello world")
        hhaxlow werld


## Licence

**Copyright 2015-2020 Mathieu Bernard**

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
