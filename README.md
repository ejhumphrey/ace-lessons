# ace-lessons
Four Timely Lessons on Automatic Chord Estimation

# Installation

This work was completed during the active development of a few projects. As such, getting these corresponding projects in their respective states requires a tiny bit of wrangling.

## ace-lessons
As a first step, you'll want to clone (or download) this repository. This is most easily achieved with git on the commandline:

`git clone git://github.com/ejhumphrey/ace-lessons.git ace-lessons`

From here on, this `ace-lessons` directory will be referred to as `REPO` for any bash commands.

## JAMS
The data and demo notebook depend on an older and less robust version of the JAMS library, v0.1, which preceded a proper setup file. In order to travel back in time to this previous state, you'll need to execute the following:

```
git clone git://github.com/marl/jams.git jams-dev
cd jams-dev
git checkout tags/v0.1
```

For the demonstration code to see the library, you'll have to do one of the following:

* Move `pyjams` into `REPO`
* Add it to your python path via `export PYTHONPATH=`pwd`:$PYTHONPATH`
* Move / symlink it alongside other python libraries

## Installation
More details coming soon!


