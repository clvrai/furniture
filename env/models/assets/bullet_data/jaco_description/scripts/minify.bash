#!/bin/bash

# go to the meshes folder
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MESHES="$DIR/../meshes"

# remove old minified files
cd $MESHES
rm *.min.dae

# re-minify
for f in *.dae ; do xmllint --noblanks $f > "${f/.dae/.min.dae}" ; done
